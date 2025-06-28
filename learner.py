import argparse
import os
import pickle
import time
import threading

os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")
from typing import Set
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp

from drop_stack_ai.utils.device_info import print_device_info
from drop_stack_ai.model.network import DropStackNet, create_model
from drop_stack_ai.training.train import (
    TrainConfig,
    make_update_fn,
    pmap_update_fn,
    create_train_state,
)
from drop_stack_ai.training.data_loader import data_loader
from drop_stack_ai.training.replay_buffer import ReplayBuffer
from drop_stack_ai.utils.serialization import (
    load_params,
    save_params,
    load_bytes,
)
from google.cloud import storage


DEFAULT_BUCKET = os.environ.get("DROPSTACK_BUCKET", "gs://drop-stack-ai-data-12345")
DEFAULT_MODEL = os.path.join(DEFAULT_BUCKET, "checkpoints", "model.msgpack")
DEFAULT_EPISODES = os.path.join(DEFAULT_BUCKET, "episodes")


def load_buffer_bytes(data: bytes) -> ReplayBuffer:
    obj = pickle.loads(data)
    if isinstance(obj, ReplayBuffer):
        return obj
    if isinstance(obj, list):
        return ReplayBuffer(data=obj)
    raise TypeError("Unrecognized buffer format")


def list_files(path: str) -> list[str]:
    if path.startswith("gs://"):
        bucket, prefix = path[5:].split("/", 1)
        client = storage.Client()
        return [
            f"gs://{bucket}/{blob.name}"
            for blob in client.list_blobs(bucket, prefix=prefix)
        ]
    else:
        return [os.path.join(path, f) for f in os.listdir(path)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Learner")
    parser.add_argument(
        "--data",
        type=str,
        default=DEFAULT_EPISODES,
        help="Episode file directory or gs:// bucket",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Path to save model parameters",
    )
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--save-every", type=int, default=300)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--scan-every", type=int, default=10, help="Seconds between polling for new data")
    parser.add_argument(
        "--max-scan-interval",
        type=int,
        default=300,
        help="Maximum backoff interval when polling GCS",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=4,
        help="Parallel download threads for episode files",
    )
    parser.add_argument("--log-interval", type=int, default=100, help="Steps between metric logs")
    parser.add_argument("--init-episodes", type=int, default=0, help="Generate this many episodes locally if buffer is empty")
    parser.add_argument("--processes", type=int, default=1, help="Processes for seeding episodes")
    parser.add_argument("--greedy-after", type=int, default=10, help="Steps before greedy play during seeding")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("JAX_TRACEBACK_FILTERING", "off")
    print_device_info()

    rng = jax.random.PRNGKey(args.seed)
    config = TrainConfig(
        batch_size=args.batch_size,
        steps=args.steps,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        checkpoint_path=args.model,
        buffer_size=args.buffer_size,
        mixed_precision=args.mixed_precision,
    )
    state, model = create_train_state(rng, config)

    devices = jax.local_devices()
    n_devices = len(devices)
    if n_devices > 1:
        state = jax.device_put_replicated(state, devices)
        update_fn = pmap_update_fn(model)
    else:
        update_fn = jax.jit(make_update_fn(model))

    buffer = ReplayBuffer(max_episodes=args.buffer_size)
    if args.init_episodes > 0:
        print(f"[learner] generating {args.init_episodes} seed episodes")
        tmp = ReplayBuffer()
        rng = self_play_parallel(
            model,
            state.params,
            rng,
            tmp,
            episodes=args.init_episodes,
            processes=args.processes,
            greedy_after=args.greedy_after,
        )
        buffer.extend(tmp)
        print(f"[learner] initial buffer size={len(buffer)}")
    processed: Set[str] = set()
    last_save = time.time()
    loader = data_loader(buffer, args.batch_size, devices=devices, prefetch=2)

    stop_event = threading.Event()

    def _scan() -> None:
        interval = args.scan_every
        is_gcs = args.data.startswith("gs://")
        with ThreadPoolExecutor(max_workers=args.download_workers) as executor:
            while not stop_event.is_set():
                paths = [p for p in list_files(args.data) if p not in processed]
                if paths:
                    futures = {executor.submit(load_bytes, p): p for p in paths}
                    for fut in futures:
                        path = futures[fut]
                        try:
                            data = fut.result()
                            new_buf = load_buffer_bytes(data)
                            buffer.extend(new_buf)
                            processed.add(path)
                            print(
                                f"[learner] loaded data from {path}, buffer size={len(buffer)}"
                            )
                        except Exception as e:
                            print(f"[learner] failed to load {path}: {e}")
                    interval = args.scan_every
                else:
                    if is_gcs:
                        interval = min(interval * 2, args.max_scan_interval)
                time.sleep(interval)

    threading.Thread(target=_scan, daemon=True).start()

    step = 0
    try:
        while True:
            if len(buffer) == 0:
                time.sleep(0.1)
                continue

            batch = next(loader)
            state, metrics = update_fn(state, batch)
            step += 1

            if step % args.log_interval == 0:
                metrics = jax.tree_util.tree_map(lambda x: float(jnp.mean(x)), metrics)
                print(
                    f"step {step}: loss={metrics['loss']:.4f} "
                    f"policy_loss={metrics['policy_loss']:.4f} value_loss={metrics['value_loss']:.4f}"
                )

            now = time.time()
            if now - last_save >= args.save_every:
                params = state.params
                if n_devices > 1:
                    params = jax.tree_util.tree_map(lambda x: x[0], params)
                params = jax.device_get(params)
                save_params(params, args.model)
                print(f"[learner] saved model to {args.model}")
                last_save = now
    finally:
        stop_event.set()


if __name__ == "__main__":
    main()
