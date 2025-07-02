from __future__ import annotations

from typing import Any, Dict

import jax
import jax.numpy as jnp


def state_to_arrays(
    state: Dict[str, Any],
    *,
    device: jax.Device | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Convert env state dict into arrays for the model.

    Parameters
    ----------
    state:
        Environment state dictionary returned by ``DropStackEnv.get_state``.
    device:
        Optional JAX device to place the resulting arrays on. When ``None`` the
        current default device is used.
    """
    # Pad each column to the fixed ``MAX_HEIGHT`` and build the board in one go.
    board_list = [
        col + [0] * (6 - len(col)) if len(col) < 6 else col
        for col in state["board"]
    ]
    board = jnp.array(board_list, dtype=jnp.float32)
    current = jnp.array(state["current_tile"], dtype=jnp.float32)
    next_tile = jnp.array(state["next_tile"], dtype=jnp.float32)
    # Ensure arrays reside on the desired device for jitted functions
    board = jax.device_put(board, device)
    current = jax.device_put(current, device)
    next_tile = jax.device_put(next_tile, device)
    return board, current, next_tile
