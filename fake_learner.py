import argparse
import os
import time

os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")
os.environ.setdefault("JAX_TRACEBACK_FILTERING", "off")

import jax
import jax.numpy as jnp

from drop_stack_ai.utils.device_info import print_device_info
from drop_stack_ai.model.network import create_model
from drop_stack_ai.training.train import (
    TrainConfig,
    create_train_state,
    make_update_fn,
    pmap_update_fn,
)


def random_batch(rng: jax.random.PRNGKey, batch_size: int, devices: list[jax.Device]):
    n_devices = len(devices)
    rngs = jax.random.split(rng, 5)
    board = jax.random.randint(rngs[0], (batch_size, 5, 6), 0, 12, dtype=jnp.float32)
    current = jax.random.randint(rngs[1], (batch_size,), 0, 12, dtype=jnp.float32)
    nxt = jax.random.randint(rngs[2], (batch_size,), 0, 12, dtype=jnp.float32)
    policy = jax.random.uniform(rngs[3], (batch_size, 5), dtype=jnp.float32)
    policy = policy / policy.sum(axis=1, keepdims=True)
    value = jax.random.uniform(rngs[4], (batch_size,), dtype=jnp.float32)
    batch = {
        "board": board,
        "current": current,
        "next": nxt,
        "policy": policy,
        "value": value,
    }
    if n_devices > 1:
        per_dev = batch_size // n_devices
        batch = {k: v.reshape((n_devices, per_dev) + v.shape[1:]) for k, v in batch.items()}
        batch = {k: jax.device_put_sharded(list(v), devices) for k, v in batch.items()}
    else:
        batch = jax.device_put(batch, devices[0])
    return batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark training with fake data")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print_device_info()

    rng = jax.random.PRNGKey(args.seed)
    config = TrainConfig(
        batch_size=args.batch_size,
        steps=args.steps,
        hidden_size=args.hidden_size,
        mixed_precision=args.mixed_precision,
        checkpoint_path=None,
    )
    state, model = create_train_state(rng, config)
    if args.mixed_precision:
        jax.config.update("jax_default_matmul_precision", "float16")

    devices = jax.local_devices()
    update_fn = pmap_update_fn(model) if len(devices) > 1 else jax.jit(make_update_fn(model))

    times_ms = []
    for step in range(1, args.steps + 1):
        rng, batch_rng = jax.random.split(rng)
        batch = random_batch(batch_rng, args.batch_size, devices)
        t0 = time.perf_counter()
        state, metrics = update_fn(state, batch)
        jax.block_until_ready(metrics["loss"])
        dt = (time.perf_counter() - t0) * 1000
        times_ms.append(dt)
        if step == 1 or step == args.steps or step % 10 == 0:
            m = jax.tree_util.tree_map(float, metrics)
            print(f"step {step}: loss={m['loss']:.4f} ({dt:.2f} ms)")

    avg = sum(times_ms) / len(times_ms)
    print(f"average step: {avg:.2f} ms over {args.steps} steps")


if __name__ == "__main__":
    main()
