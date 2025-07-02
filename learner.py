# learner.py   (instrumented)
import argparse, os, pickle, time, threading
from typing import Set
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_gpu_triton_gemm_any=True "
    "--xla_gpu_autotune_level=2 "
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=16",
)

import jax, jax.numpy as jnp

from drop_stack_ai.utils.device_info import print_device_info
from drop_stack_ai.model.network import create_model, DropStackNet
from drop_stack_ai.training.train import (
    TrainConfig,
    make_update_fn,
    pmap_update_fn,
    create_train_state,
)
from drop_stack_ai.training.data_loader import data_loader
from drop_stack_ai.training.replay_buffer import ReplayBuffer
from drop_stack_ai.utils.serialization import load_params, save_params, load_bytes
from google.cloud import storage

# ----------------------------------------------------------------------------- #
#  Configuration constants                                                      #
# ----------------------------------------------------------------------------- #
DEFAULT_BUCKET   = os.environ.get("DROPSTACK_BUCKET", "gs://drop-stack-ai-data-12345")
DEFAULT_MODEL    = f"{DEFAULT_BUCKET}/checkpoints/model.msgpack"
DEFAULT_LATEST   = f"{DEFAULT_BUCKET}/checkpoints/model_latest.msgpack"
DEFAULT_EPISODES = f"{DEFAULT_BUCKET}/episodes"

# ----------------------------------------------------------------------------- #
#  Helpers (unchanged except minor formatting)                                  #
# ----------------------------------------------------------------------------- #
def load_buffer_bytes(data: bytes) -> ReplayBuffer:
    obj = pickle.loads(data)
    if isinstance(obj, ReplayBuffer):
        return obj
    if isinstance(obj, list):
        return ReplayBuffer(data=obj)
    raise TypeError("Unrecognized buffer format")

def list_files(path: str, client: storage.Client | None = None) -> list[str]:
    if path.startswith("gs://"):
        bucket, prefix = path[5:].split("/", 1)
        client = client or storage.Client()
        return [f"gs://{bucket}/{b.name}" for b in client.list_blobs(bucket, prefix=prefix)]
    return [os.path.join(path, f) for f in os.listdir(path)]

def make_fused_update(update_fn, n_steps: int):
    def fused(state, loader):
        metrics_sum = None
        for _ in range(n_steps):
            batch = next(loader)                        # ---- waits here if buffer empty
            state, metrics = update_fn(state, batch)    # ---- GPU work
            metrics_sum = metrics if metrics_sum is None else jax.tree_util.tree_map(
                lambda a, b: a + b, metrics_sum, metrics)
        metrics_mean = jax.tree_util.tree_map(lambda x: x / n_steps, metrics_sum)
        return state, metrics_mean
    return fused

# ----------------------------------------------------------------------------- #
#  Main                                                                         #
# ----------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Learner")
    parser.add_argument("--data",  type=str, default=DEFAULT_EPISODES)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--latest-model", type=str, default=DEFAULT_LATEST)
    parser.add_argument("--hidden-size",    type=int,   default=512)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--batch-size",     type=int,   default=1024)
    parser.add_argument("--learning-rate",  type=float, default=1e-3)
    parser.add_argument("--buffer-size",    type=int,   default=500_000)
    parser.add_argument("--save-every",     type=int,   default=1000)
    parser.add_argument("--steps",          type=int,   default=10_000)
    parser.add_argument("--scan-every",     type=int,   default=5)
    parser.add_argument("--max-scan-interval", type=int, default=60)
    parser.add_argument("--download-workers",  type=int, default=8)
    parser.add_argument("--log-interval",      type=int, default=64)  # << changed
    parser.add_argument("--init-episodes",     type=int, default=0)
    parser.add_argument("--processes",         type=int, default=1)
    parser.add_argument("--greedy-after",      type=int, default=10)
    parser.add_argument("--seed",              type=int, default=0)
    parser.add_argument("-p", "--profile",     action="store_true",
                        help="Print per-iteration timing (wait vs GPU).")
    args = parser.parse_args()

    os.environ.setdefault("JAX_TRACEBACK_FILTERING", "off")
    print_device_info()

    rng = jax.random.PRNGKey(args.seed)
    config = TrainConfig(
        batch_size      = args.batch_size,
        steps           = args.steps,
        learning_rate   = args.learning_rate,
        hidden_size     = args.hidden_size,
        checkpoint_path = args.model,
        buffer_size     = args.buffer_size,
        mixed_precision = args.mixed_precision,
        upload_path     = args.latest_model,
    )
    state, model = create_train_state(rng, config)

    devices    = jax.local_devices()
    n_devices  = len(devices)
    update_fn  = pmap_update_fn(model) if n_devices > 1 else jax.jit(make_update_fn(model))

    fused_steps = int(os.getenv("FUSED_STEPS", "8"))
    update_fn   = make_fused_update(update_fn, fused_steps)

    buffer = ReplayBuffer(max_episodes=args.buffer_size)
    processed: Set[str] = set()
    loader   = data_loader(buffer, args.batch_size, devices=devices, prefetch=2)
    gcs_client = storage.Client() if args.data.startswith("gs://") else None

    # --- background GCS scan thread (unchanged) --------------------------------
    stop_event, episode_q = threading.Event(), Queue(maxsize=16)
    def _scan(client: storage.Client | None) -> None:
        with ThreadPoolExecutor(max_workers=args.download_workers) as ex:
            while not stop_event.is_set():
                paths = [p for p in list_files(args.data, client) if p not in processed]
                if paths:
                    futures = {ex.submit(load_bytes, p): p for p in paths}
                    for fut in futures:
                        pth = futures[fut]
                        try:
                            new_buf = load_buffer_bytes(fut.result())
                            episode_q.put(new_buf)
                            processed.add(pth)
                        except Exception as e:
                            print(f"[learner] failed to load {pth}: {e}")
                time.sleep(args.scan_every)
    threading.Thread(target=_scan, args=(gcs_client,), daemon=True).start()

    # ------------------------------------------------------------------------- #
    step, last_save = 0, time.time()
    try:
        while True:
            while not episode_q.empty():
                buffer.extend(episode_q.get())

            if len(buffer) == 0:
                time.sleep(0.1)
                continue

            # ---------------- profiling start ----------------------------------
            t0 = time.perf_counter()

            state, metrics = update_fn(state, loader)
            jax.block_until_ready(metrics["loss"])   # make sure GPU finished

            t1 = time.perf_counter()
            wait_s  = t1 - t0
            perf_ms = wait_s * 1e3 / fused_steps     # per mini-step

            step += fused_steps

            if args.profile and step % args.log_interval == 0:
                print(f"[profile] buffer={len(buffer):,}  "
                      f"queue={episode_q.qsize():2d}  "
                      f"fused={fused_steps}  "
                      f"avg_step {perf_ms:.2f} ms")

            # ---------------- normal logging -----------------------------------
            if step % args.log_interval == 0:
                m = jax.tree_util.tree_map(float, metrics)
                print(
                    f"[learner] step {step}/{args.steps} | "
                    f"loss {m['loss']:.4f} "
                    f"policy {m['policy_loss']:.4f} "
                    f"value {m['value_loss']:.4f}"
                )

            # ---------------- checkpointing ------------------------------------
            if (time.time() - last_save) >= args.save_every:
                params = jax.tree_util.tree_map(lambda x: x[0], state.params) if n_devices > 1 else state.params
                save_params(jax.device_get(params), args.model)
                print(f"[learner] saved model → {args.model}")
                if args.latest_model:
                    save_params(params, args.latest_model)
                    print(f"[learner] uploaded latest → {args.latest_model}")
                last_save = time.time()
    finally:
        stop_event.set()

if __name__ == "__main__":
    main()
