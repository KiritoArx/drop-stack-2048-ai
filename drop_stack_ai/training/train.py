from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax

from drop_stack_ai.model.network import DropStackNet, create_model
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


# -----------------------------------------------------------------------------
# Training state
# -----------------------------------------------------------------------------

@dataclass
class TrainConfig:
    batch_size: int = 32
    steps: int = 1000
    learning_rate: float = 1e-3
    hidden_size: int = 128


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

        if step % 10 == 0:
            # Metrics may be replicated
            metrics = jax.tree_map(lambda x: float(jnp.mean(x)), metrics)
            print(
                f"step {step}: loss={metrics['loss']:.4f} "
                f"policy_loss={metrics['policy_loss']:.4f} value_loss={metrics['value_loss']:.4f}"
            )


if __name__ == "__main__":
    buffer = ReplayBuffer()
    # minimal dummy data to allow script to run
    env_state = {
        "board": [[] for _ in range(5)],
        "current_tile": 2,
        "next_tile": 2,
        "score": 0,
        "done": False,
    }
    dummy_policy = jnp.ones(5, dtype=jnp.float32) / 5
    buffer.add_episode([env_state], [dummy_policy], [0.0])
    config = TrainConfig(batch_size=1, steps=5)
    train(buffer, config=config)
