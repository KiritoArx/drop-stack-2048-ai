from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import os
import argparse
import pickle

os.environ.setdefault("JAX_TRACEBACK_FILTERING", "off")

import jax
import jax.numpy as jnp
from flax.training import train_state
import optax

from drop_stack_ai.model.network import DropStackNet, create_model
from drop_stack_ai.selfplay.self_play import self_play
from .replay_buffer import ReplayBuffer
from drop_stack_ai.utils.serialization import load_params, save_params
from drop_stack_ai.utils.state_utils import state_to_arrays


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _prepare_batch(samples: list[Dict[str, Any]]) -> Dict[str, jnp.ndarray]:
    boards = []
    currents = []
    nexts = []
    policies = []
    values = []
    for item in samples:
        b, c, n = state_to_arrays(item["state"])
        boards.append(b)
        currents.append(c)
        nexts.append(n)
        policy = jnp.array(item["policy"], dtype=jnp.float32)
        if policy.ndim == 1 and policy.sum() != 0:
            # Ensure policy represents a probability distribution. The
            # replay buffer may store raw visit counts from MCTS.
            policy = policy / policy.sum()
        policies.append(policy)

        # ``value`` stores ``log(final_score + 1)`` for the episode.
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
    checkpoint_path: Optional[str] = None


def create_train_state(
    rng: jax.random.PRNGKey, config: TrainConfig
) -> train_state.TrainState:
    model, params = create_model(rng, hidden_size=config.hidden_size)
    if config.checkpoint_path and os.path.exists(config.checkpoint_path):
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
            batch = {
                k: v.reshape((n_devices, per_dev) + v.shape[1:])
                for k, v in batch.items()
            }
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
        "--steps", type=int, default=1000, help="Number of training steps"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument(
        "--hidden-size", type=int, default=128, help="Model hidden size"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join("checkpoints", "model.msgpack"),
        help="Path to save or load model parameters",
    )
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
            rng = self_play(model, params, rng, buffer, greedy_after=args.greedy_after)

    if len(buffer) == 0:
        raise ValueError("Replay buffer is empty")

    config = TrainConfig(
        batch_size=args.batch_size,
        steps=args.steps,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        checkpoint_path=args.checkpoint,
    )
    train(buffer, seed=args.seed, config=config)
