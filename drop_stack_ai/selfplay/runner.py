from __future__ import annotations

from typing import List, Dict, Any

import math

import jax
import jax.numpy as jnp

from drop_stack_ai.env.drop_stack_env import DropStackEnv
from drop_stack_ai.model.network import DropStackNet
from drop_stack_ai.training.replay_buffer import ReplayBuffer


def _state_to_arrays(state: Dict[str, Any]) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Convert a raw env state dict into arrays usable by the model."""
    board = jnp.zeros((5, 6), dtype=jnp.float32)
    for c, col in enumerate(state["board"]):
        if col:
            board = board.at[c, : len(col)].set(jnp.array(col, dtype=jnp.float32))
    current = jnp.array(state["current_tile"], dtype=jnp.float32)
    next_tile = jnp.array(state["next_tile"], dtype=jnp.float32)
    return board, current, next_tile


def self_play(
    model: DropStackNet,
    params: Dict[str, Any],
    rng: jax.random.PRNGKey,
    buffer: ReplayBuffer,
    *,
    greedy: bool = False,
) -> jax.random.PRNGKey:
    """Play one episode and store the experience in ``buffer``."""
    env = DropStackEnv()
    states: List[Dict[str, Any]] = []
    policies: List[jnp.ndarray] = []
    values: List[float] = []

    done = False
    while not done:
        raw_state = env.get_state()
        board, current, next_tile = _state_to_arrays(raw_state)
        logits, value_pred = model.apply(params, board, current, next_tile)
        policy = jax.nn.softmax(logits)

        if greedy:
            action = int(jnp.argmax(policy))
        else:
            rng, key = jax.random.split(rng)
            action = int(jax.random.choice(key, 5, p=policy))

        states.append(raw_state)
        policies.append(policy)
        values.append(0.0)  # placeholder

        _, _, done = env.step(action)

    # Episode finished, assign final score as the value target
    final_score = math.log(env.score + 1)
    values = [float(final_score)] * len(values)
    buffer.add_episode(states, policies, values)
    return rng
