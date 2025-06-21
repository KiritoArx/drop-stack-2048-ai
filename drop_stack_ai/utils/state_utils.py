from __future__ import annotations

from typing import Any, Dict

import jax
import jax.numpy as jnp


def state_to_arrays(
    state: Dict[str, Any],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Convert env state dict into arrays for the model."""
    board = jnp.zeros((5, 6), dtype=jnp.float32)
    for c, col in enumerate(state["board"]):
        if col:
            board = board.at[c, : len(col)].set(jnp.array(col, dtype=jnp.float32))
    current = jnp.array(state["current_tile"], dtype=jnp.float32)
    next_tile = jnp.array(state["next_tile"], dtype=jnp.float32)
    # Ensure arrays reside on device for jitted functions
    board = jax.device_put(board)
    current = jax.device_put(current)
    next_tile = jax.device_put(next_tile)
    return board, current, next_tile
