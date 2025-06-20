from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import os
import argparse
import pickle

os.environ.setdefault("JAX_TRACEBACK_FILTERING", "off")

import jax
import jax.numpy as jnp
from flax.training import train_state
import optax

from drop_stack_ai.model.network import DropStackNet, create_model
from drop_stack_ai.selfplay.runner import self_play
from .replay_buffer import ReplayBuffer


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _state_to_arrays(state: Dict[str, Any]) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Convert env state dict into model input arrays."""
    board = jnp.zeros((5, 6), dtype=jnp.float32)
    for c, col in enumerate(state["board"]):
        if col:
            board = board.at[c, : len(col)].set(jnp.array(col, dtype=jnp.float32))
    current = jnp.array(state["current_tile"], dtype=jnp.float32)
    next_tile = jnp.array(state["next_tile"], dtype=jnp.float32)
    return board, current, next_tile


def _prepare_batch(samples: list[Dict[str, Any]]) -> Dict[str, jnp.ndarray]:
    boards = []
    currents = []
    nexts = []
    policies = []
    values = []
    for item in samples:
        b, c, n = _state_to_arrays(item["state"])
        boards.append(b)
        currents.append(c)
        nexts.append(n)
        policies.append(jnp.array(item["policy"], dtype=jnp.float32))
        values.append(jnp.array(item["value"], dtype=jnp.float32))
    batch = {
        "board": jnp.stack(boards),
        "current": jnp.stack(currents),
        "next": jnp.stack(nexts),
        "policy": jnp.stack(policies),
        "value": jnp.stack(values),
    }
    return batch


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
    batch_size: int = 32
    steps: int = 1000
    learning_rate: float = 1e-3
    hidden_size: int = 128
    log_interval: int = 10


def create_train_state(rng: jax.random.PRNGKey, config: TrainConfig) -> train_state.TrainState:
    model, params = create_model(rng, hidden_size=config.hidden_size)
    tx = optax.adam(config.learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx), model


# -----------------------------------------------------------------------------
# Training step
# -----------------------------------------------------------------------------

def make_update_fn(model: DropStackNet):
    def loss_fn(params, batch):
        logits, value_pred = model.apply(params, batch["board"], batch["current"], batch["next"])
        policy_loss = optax.softmax_cross_entropy(logits, batch["policy"]).mean()
        value_loss = jnp.mean((value_pred - batch["value"]) ** 2)
        loss = policy_loss + value_loss
        return loss, (policy_loss, value_loss)

    def train_step(state: train_state.TrainState, batch: Dict[str, jnp.ndarray]):
        (loss, (p_loss, v_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, batch)
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

def train(buffer: ReplayBuffer, *, seed: int = 0, config: TrainConfig | None = None) -> None:
    if config is None:
        config = TrainConfig()

    rng = jax.random.PRNGKey(seed)
    state, model = create_train_state(rng, config)

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

    for step in range(1, config.steps + 1):
        samples = buffer.sample(config.batch_size)
        batch = _prepare_batch(samples)

        if n_devices > 1:
            # reshape batch for pmapping
            per_dev = config.batch_size // n_devices
            batch = {k: v.reshape((n_devices, per_dev) + v.shape[1:]) for k, v in batch.items()}
        state, metrics = update_fn(state, batch)

        if step == 1 or step == config.steps or step % config.log_interval == 0:
            # Metrics may be replicated
            metrics = jax.tree_util.tree_map(lambda x: float(jnp.mean(x)), metrics)
            print(
                f"step {step}: loss={metrics['loss']:.4f} "
                f"policy_loss={metrics['policy_loss']:.4f} value_loss={metrics['value_loss']:.4f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Drop Stack 2048 model")
    parser.add_argument("--buffer", type=str, help="Path to replay buffer pickle")
    parser.add_argument("--self-play", type=int, default=0, help="Generate this many self-play episodes before training")
    parser.add_argument("--steps", type=int, default=1000, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-size", type=int, default=128, help="Model hidden size")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed")
    args = parser.parse_args()

    if args.buffer:
        buffer = load_buffer(args.buffer)
    else:
        buffer = ReplayBuffer()

    rng = jax.random.PRNGKey(args.seed)
    if args.self_play > 0:
        model, params = create_model(rng, hidden_size=args.hidden_size)
        for _ in range(args.self_play):
            rng = self_play(model, params, rng, buffer)

    if len(buffer) == 0:
        raise ValueError("Replay buffer is empty")

    config = TrainConfig(
        batch_size=args.batch_size,
        steps=args.steps,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
    )
    train(buffer, seed=args.seed, config=config)
