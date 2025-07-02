import argparse
import os
import pickle
import time
import threading

os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_gpu_triton_gemm_any=True "
    "--xla_gpu_autotune_level=2 "
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=16",
)
from typing import Set
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

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
DEFAULT_LATEST = os.path.join(DEFAULT_BUCKET, "checkpoints", "model_latest.msgpack")
DEFAULT_EPISODES = os.path.join(DEFAULT_BUCKET, "episodes")


def load_buffer_bytes(data: bytes) -> ReplayBuffer:
    obj = pickle.loads(data)
    if isinstance(obj, ReplayBuffer):
        return obj
    if isinstance(obj, list):
        return ReplayBuffer(data=obj)
    raise TypeError("Unrecognized buffer format")


def list_files(path: str, client: storage.Client | None = None) -> list[str]:
    """Return a list of file paths contained in ``path``.

    If ``path`` is a ``gs://`` URI, ``client`` is used for the storage
    operations. When ``client`` is ``None`` a new ``storage.Client`` will be
    created on demand. Local paths simply list directory contents.
    """

    if path.startswith("gs://"):
        bucket, prefix = path[5:].split("/", 1)
        if client is None:
            client = storage.Client()
        return [
            f"gs://{bucket}/{blob.name}"
            for blob in client.list_blobs(bucket, prefix=prefix)
        ]
    else:
        return [os.path.join(path, f) for f in os.listdir(path)]


def make_fused_update(update_fn, n_steps: int):
    """Wrap ``update_fn`` to perform multiple steps per call.

    The loader itself is passed to the returned function so it can pull
    consecutive batches without returning to Python between steps. Metrics are
    accumulated and averaged over ``n_steps``.
    """

    def fused(state, loader):
        metrics_sum = None
        for _ in range(n_steps):
            batch = next(loader)
            state, metrics = update_fn(state, batch)
            if metrics_sum is None:
                metrics_sum = metrics
            else:
                metrics_sum = jax.tree_util.tree_map(
                    lambda a, b: a + b, metrics_sum, metrics
                )

        metrics_mean = jax.tree_util.tree_map(lambda x: x / n_steps, metrics_sum)
        return state, metrics_mean

    return fused


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
    parser.add_argument(
        "--latest-model",
        type=str,
        default=DEFAULT_LATEST,
        help="gs:// path for the continuously updated model",
    )
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument(
        "--batch-size", type=int, default=1024, help="Maximizes GPU usage."
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="More stable for large batches.",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=500_000,
        help="Keeps richer replay data available.",
    )
    parser.add_argument(
        "--save-every", type=int, default=1000, help="Checkpoint every 1000 steps."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10000,
        help="Fewer interruptions, longer training cycles.",
    )
    parser.add_argument(
        "--scan-every", type=int, default=5, help="Frequent scanning for new episodes."
    )
    parser.add_argument(
        "--max-scan-interval",
        type=int,
        default=60,
        help="Quickly detect newly uploaded data.",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=8,
        help="Parallel downloads for episodes.",
    )
    parser.add_argument(
        "--log-interval", type=int, default=8, help="Reduce logging overhead."
    )
    parser.add_argument(
        "--init-episodes",
        type=int,
        default=0,
        help="Generate this many episodes locally if buffer is empty",
    )
    parser.add_argument(
        "--processes", type=int, default=1, help="Processes for seeding episodes"
    )
    parser.add_argument(
        "--greedy-after",
        type=int,
        default=10,
        help="Steps before greedy play during seeding",
    )
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
        upload_path=args.latest_model,
    )
    state, model = create_train_state(rng, config)

    devices = jax.local_devices()
    n_devices = len(devices)
    if n_devices > 1:
        state = jax.device_put_replicated(state, devices)
        update_fn = pmap_update_fn(model)
    else:
        update_fn = jax.jit(make_update_fn(model))

    fused_steps = int(os.getenv("FUSED_STEPS", "8"))
    update_fn = make_fused_update(update_fn, fused_steps)

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

    gcs_client = storage.Client() if args.data.startswith("gs://") else None

    stop_event = threading.Event()
    episode_q: Queue = Queue(maxsize=16)

    def _scan(client: storage.Client | None) -> None:
        with ThreadPoolExecutor(max_workers=args.download_workers) as executor:
            while not stop_event.is_set():
                paths = [p for p in list_files(args.data, client) if p not in processed]
                if paths:
                    futures = {executor.submit(load_bytes, p): p for p in paths}
                    for fut in futures:
                        path = futures[fut]
                        try:
                            data = fut.result()
                            new_buf = load_buffer_bytes(data)
                            episode_q.put(new_buf)
                            processed.add(path)
                        except Exception as e:
                            print(f"[learner] failed to load {path}: {e}")
                time.sleep(args.scan_every)

    threading.Thread(target=_scan, args=(gcs_client,), daemon=True).start()

    step = 0
    try:
        while True:
            while not episode_q.empty():
                buffer.extend(episode_q.get())

            if len(buffer) == 0:
                time.sleep(0.1)
                continue

            state, metrics = update_fn(state, loader)
            step += fused_steps

            if step % args.log_interval == 0:
                metrics = jax.tree_util.tree_map(lambda x: float(jnp.mean(x)), metrics)
                print(
                    f"[learner] step {step}/{args.steps} | loss: {metrics['loss']:.4f} "
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
                if args.latest_model:
                    save_params(params, args.latest_model)
                    print(f"[learner] uploaded model to {args.latest_model}")
                last_save = now
    finally:
        stop_event.set()


if __name__ == "__main__":
    main()
