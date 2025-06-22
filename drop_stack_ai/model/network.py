from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn


class DropStackNet(nn.Module):
    """Simple policy and value network for Drop Stack 2048."""

    hidden_size: int = 128
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, board: jnp.ndarray, current_tile: jnp.ndarray, next_tile: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass.

        Args:
            board: array of shape ``(5, 6)`` or ``(batch, 5, 6)`` with tile values.
            current_tile: scalar array (or ``(batch,)``) with the current tile value.
            next_tile: scalar array (or ``(batch,)``) with the next tile value.

        Returns:
            Tuple of ``(policy_logits, value)`` where ``policy_logits`` are the
            unnormalised action preferences and ``value`` is a scalar evaluation
            of the position. These outputs are consumed directly by the MCTS
            implementation.
        """
        # Flatten the board and take log2 encoding to keep values in a reasonable range.
        if board.ndim == 3:
            # Batched input
            board_flat = jnp.log2(jnp.maximum(board, 1)).reshape(board.shape[0], -1)
            tile_feats = jnp.log2(
                jnp.maximum(jnp.stack([current_tile, next_tile], axis=1), 1)
            )
            x = jnp.concatenate([board_flat, tile_feats], axis=1)
        else:
            board_flat = jnp.log2(jnp.maximum(board, 1)).reshape(-1)
            tile_feats = jnp.log2(jnp.maximum(jnp.stack([current_tile, next_tile]), 1))
            x = jnp.concatenate([board_flat, tile_feats], axis=0)

        x = nn.Dense(self.hidden_size, dtype=self.dtype)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_size, dtype=self.dtype)(x)
        x = nn.relu(x)

        # Two separate heads: policy provides logits for MCTS priors, value
        # produces a scalar evaluation of the current position.
        policy_logits = nn.Dense(5, dtype=self.dtype)(x)
        value = nn.Dense(1, dtype=self.dtype)(x)
        value = value.squeeze(axis=-1)
        return policy_logits.astype(jnp.float32), value.astype(jnp.float32)


def create_model(
    rng: jax.random.PRNGKey,
    *,
    hidden_size: int = 128,
    dtype: jnp.dtype = jnp.float32,
) -> tuple[DropStackNet, dict]:
    """Utility to create the model and initialise its parameters on the default device."""
    model = DropStackNet(hidden_size=hidden_size, dtype=dtype)
    dummy_board = jnp.zeros((5, 6), jnp.float32)
    dummy_tile = jnp.array(2, jnp.float32)
    params = model.init(rng, dummy_board, dummy_tile, dummy_tile)
    # Ensure parameters live on device (TPU compatible)
    params = jax.device_put(params)
    return model, params


