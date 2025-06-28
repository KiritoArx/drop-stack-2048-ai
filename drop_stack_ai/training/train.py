from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import os
import argparse
import pickle
import time

os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")
os.environ.setdefault("JAX_TRACEBACK_FILTERING", "off")

import jax
import jax.numpy as jnp
from flax.training import train_state
import optax

from drop_stack_ai.utils.device_info import print_device_info

from drop_stack_ai.model.network import DropStackNet, create_model
from drop_stack_ai.selfplay.self_play import (
    self_play,
    self_play_parallel,
    launch_self_play_workers,
    launch_self_play_workers_dynamic,
)
from .data_loader import data_loader
from .replay_buffer import ReplayBuffer
from drop_stack_ai.utils.serialization import load_params, save_params
from drop_stack_ai.utils.serialization import load_bytes
from google.cloud import storage




def load_buffer(path: str) -> ReplayBuffer:
    """Load a ``ReplayBuffer`` from ``path``."""
    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, ReplayBuffer):
        return data
    if isinstance(data, list):
        return ReplayBuffer(data=data)
    raise TypeError("Unrecognized buffer format")


# -----------------------------------------------------------------------------
# Training state
# -----------------------------------------------------------------------------


@dataclass
class TrainConfig:
    batch_size: int = 512
    steps: int = 100_000
    learning_rate: float = 2e-3
    hidden_size: int = 1024
    log_interval: int = 10
    checkpoint_path: Optional[str] = None
    workers: int = 0
    buffer_size: int = 200_000
    greedy_after: int | None = 10
    mixed_precision: bool = False
    upload_path: Optional[str] = None


def create_train_state(
    rng: jax.random.PRNGKey, config: TrainConfig
) -> train_state.TrainState:
    dtype = jnp.float16 if config.mixed_precision else jnp.float32
    model, params = create_model(rng, hidden_size=config.hidden_size, dtype=dtype)
    if config.checkpoint_path:
        exists = False
        if config.checkpoint_path.startswith("gs://"):
            bucket, blob_name = config.checkpoint_path[5:].split("/", 1)
            client = storage.Client()
            blob = client.bucket(bucket).blob(blob_name)
            exists = blob.exists()
        else:
            exists = os.path.exists(config.checkpoint_path)
        if exists:
            print(f"Loading checkpoint from {config.checkpoint_path}")
            params = load_params(config.checkpoint_path, params)
    tx = optax.adam(config.learning_rate)
    return (
        train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx),
        model,
    )


# -----------------------------------------------------------------------------
# Training step
# -----------------------------------------------------------------------------


def make_update_fn(model: DropStackNet):
    def loss_fn(params, batch):
        logits, value_pred = model.apply(
            params, batch["board"], batch["current"], batch["next"]
        )
        # ``batch['policy']`` contains the MCTS visit probability distribution.
        policy_loss = optax.softmax_cross_entropy(logits, batch["policy"]).mean()
        # Values are stored as ``log(final_score + 1)`` for each step.
        value_loss = jnp.mean((value_pred - batch["value"]) ** 2)
        loss = policy_loss + value_loss
        return loss, (policy_loss, value_loss)

    def train_step(state: train_state.TrainState, batch: Dict[str, jnp.ndarray]):
        (loss, (p_loss, v_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, batch
        )
        state = state.apply_gradients(grads=grads)
        metrics = {
            "loss": loss,
            "policy_loss": p_loss,
            "value_loss": v_loss,
        }
        return state, metrics

    return train_step


def pmap_update_fn(model: DropStackNet):
    update_fn = make_update_fn(model)

    @jax.pmap
    def _step(state: train_state.TrainState, batch: Dict[str, jnp.ndarray]):
        state, metrics = update_fn(state, batch)
        return state, metrics

    return _step


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------


def train(
    buffer: ReplayBuffer, *, seed: int = 0, config: TrainConfig | None = None
) -> None:
    if config is None:
        config = TrainConfig()

    rng = jax.random.PRNGKey(seed)
    print_device_info()
    sp_rng, rng = jax.random.split(rng)
    state, model = create_train_state(rng, config)
    if config.mixed_precision:
        jax.config.update("jax_default_matmul_precision", "float16")

    devices = jax.local_devices()
    n_devices = len(devices)
    print(f"Using {n_devices} device(s) for training")

    if config.batch_size < n_devices or config.batch_size % n_devices != 0:
        print(
            "Warning: batch size does not align with available devices; "
            "falling back to single-device mode"
        )
        devices = [devices[0]]
        n_devices = 1

    if n_devices > 1:
        state = jax.device_put_replicated(state, devices)
        update_fn = pmap_update_fn(model)
    else:
        update_fn = jax.jit(make_update_fn(model))

    sp_stop = None
    if config.workers > 0:
        def _get_params() -> Dict[str, Any]:
            params = state.params
            if n_devices > 1:
                params = jax.tree_util.tree_map(lambda x: x[0], params)
            return jax.device_get(params)

        sp_stop = launch_self_play_workers_dynamic(
            model,
            _get_params,
            sp_rng,
            buffer,
            workers=config.workers,
            greedy_after=config.greedy_after,
        )
        # Give workers time to populate the buffer
        min_size = max(1, config.batch_size // 4)
        while len(buffer) < min_size:
            time.sleep(0.1)

    loader = data_loader(buffer, config.batch_size, devices=devices, prefetch=2)

    for step in range(1, config.steps + 1):
        batch = next(loader)
        state, metrics = update_fn(state, batch)

        if step == 1 or step == config.steps or step % config.log_interval == 0:
            # Metrics may be replicated
            metrics = jax.tree_util.tree_map(lambda x: float(jnp.mean(x)), metrics)
            print(
                f"step {step}: loss={metrics['loss']:.4f} "
                f"policy_loss={metrics['policy_loss']:.4f} value_loss={metrics['value_loss']:.4f}"
            )

    # Save checkpoint when training completes
    if config.checkpoint_path:
        params = state.params
        if n_devices > 1:
            params = jax.tree_util.tree_map(lambda x: x[0], params)
        params = jax.device_get(params)
        save_params(params, config.checkpoint_path)
        print(f"Saved checkpoint to {config.checkpoint_path}")
        if config.upload_path:
            save_params(params, config.upload_path)
            print(f"Uploaded checkpoint to {config.upload_path}")

    if sp_stop is not None:
        sp_stop.set()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Drop Stack 2048 model")
    parser.add_argument("--buffer", type=str, help="Path to replay buffer pickle")
    parser.add_argument(
        "--self-play",
        type=int,
        default=0,
        help="Generate this many self-play episodes before training",
    )
    parser.add_argument(
        "--greedy-after",
        type=int,
        default=10,
        help="Number of moves to sample probabilistically before switching to greedy play during self-play",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help="Number of worker processes for self-play",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Background self-play workers during training",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=200_000,
        help="Maximum episodes to store in the replay buffer",
    )
    parser.add_argument(
        "--steps", type=int, default=100_000, help="Number of training steps"
    )
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=2e-3, help="Learning rate"
    )
    parser.add_argument(
        "--hidden-size", type=int, default=1024, help="Model hidden size"
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable mixed precision training",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join("checkpoints", "model.msgpack"),
        help="Path to save or load model parameters",
    )
    parser.add_argument(
        "--upload",
        type=str,
        default=None,
        help="Optional gs:// path to upload the final model",
    )
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed")
    args = parser.parse_args()

    if args.buffer:
        buffer = load_buffer(args.buffer)
        buffer.max_episodes = args.buffer_size
    else:
        buffer = ReplayBuffer(max_episodes=args.buffer_size)

    rng = jax.random.PRNGKey(args.seed)
    if args.self_play > 0:
        dtype = jnp.float16 if args.mixed_precision else jnp.float32
        model, params = create_model(rng, hidden_size=args.hidden_size, dtype=dtype)
        if args.processes > 1:
            rng = self_play_parallel(
                model,
                params,
                rng,
                buffer,
                episodes=args.self_play,
                processes=args.processes,
                greedy_after=args.greedy_after,
            )
        else:
            for _ in range(args.self_play):
                rng = self_play(
                    model, params, rng, buffer, greedy_after=args.greedy_after
                )

    if len(buffer) == 0:
        raise ValueError("Replay buffer is empty")

    config = TrainConfig(
        batch_size=args.batch_size,
        steps=args.steps,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        checkpoint_path=args.checkpoint,
        workers=args.workers,
        buffer_size=args.buffer_size,
        greedy_after=args.greedy_after,
        mixed_precision=args.mixed_precision,
        upload_path=args.upload,
    )
    train(buffer, seed=args.seed, config=config)
