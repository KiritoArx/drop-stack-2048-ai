from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict
import math

import jax
import jax.numpy as jnp

from drop_stack_ai.env.drop_stack_env import DropStackEnv
from .network import DropStackNet


def _state_to_arrays(
    state: Dict[str, object],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Convert env state dict into arrays for the model."""
    board = jnp.zeros((5, 6), dtype=jnp.float32)
    for c, col in enumerate(state["board"]):
        if col:
            board = board.at[c, : len(col)].set(jnp.array(col, dtype=jnp.float32))
    current = jnp.array(state["current_tile"], dtype=jnp.float32)
    next_tile = jnp.array(state["next_tile"], dtype=jnp.float32)
    return board, current, next_tile


@dataclass
class Node:
    prior: float
    visit_count: int = 0
    value_sum: float = 0.0
    children: Dict[int, "Node"] = field(default_factory=dict)

    @property
    def q(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def _select_child(node: Node, c_puct: float) -> tuple[int, Node]:
    total_visits = sum(child.visit_count for child in node.children.values())
    sqrt_total = math.sqrt(total_visits + 1e-8)
    best_score = -float("inf")
    best_action = 0
    best_child = None
    for action, child in node.children.items():
        u = c_puct * child.prior * sqrt_total / (1 + child.visit_count)
        score = child.q + u
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child
    return best_action, best_child


def run_mcts(
    model: DropStackNet,
    params: Dict,
    env: DropStackEnv,
    *,
    num_simulations: int = 50,
    c_puct: float = 1.0,
) -> jnp.ndarray:
    """Run MCTS starting from ``env`` state and return a policy distribution."""
    root = Node(prior=1.0)

    for _ in range(num_simulations):
        sim_env = env.clone()
        node = root
        path: list[tuple[Node, int]] = []

        # Selection
        while node.children:
            action, child = _select_child(node, c_puct)
            sim_env.step(action)
            path.append((node, action))
            node = child
            if sim_env.done:
                break

        # Expansion and evaluation
        if sim_env.done:
            value = math.log(sim_env.score + 1)
        else:
            board, current, next_tile = _state_to_arrays(sim_env.get_state())
            logits, value_pred = model.apply(params, board, current, next_tile)
            policy = jax.nn.softmax(logits)
            if not node.children:
                for a in range(5):
                    node.children[a] = Node(float(policy[a]))
            value = float(value_pred)

        # Backup
        for parent, act in reversed(path):
            child = parent.children[act]
            child.visit_count += 1
            child.value_sum += value

    counts = jnp.array(
        [root.children[a].visit_count if a in root.children else 0 for a in range(5)],
        dtype=jnp.float32,
    )
    if counts.sum() > 0:
        policy = counts / counts.sum()
    else:
        policy = jnp.ones(5, dtype=jnp.float32) / 5
    return policy
