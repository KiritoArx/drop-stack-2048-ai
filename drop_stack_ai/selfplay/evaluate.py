from __future__ import annotations

from typing import Dict

import jax
import jax.numpy as jnp

from drop_stack_ai.env.drop_stack_env import DropStackEnv
from drop_stack_ai.model.network import DropStackNet
from drop_stack_ai.model.mcts import run_mcts


def play_game(
    model: DropStackNet,
    params: Dict,
    rng: jax.random.PRNGKey,
    *,
    simulations: int = 20,
    c_puct: float = 1.0,
    predict=None,
) -> tuple[jax.random.PRNGKey, int]:
    """Play a single game greedily and return the score."""
    env = DropStackEnv(seed=int(jax.random.randint(rng, (), 0, 2**31 - 1)))
    if predict is None:
        predict = jax.jit(model.apply)
    done = False
    while not done:
        policy = run_mcts(
            model,
            params,
            env,
            num_simulations=simulations,
            c_puct=c_puct,
            predict=predict,
        )
        action = int(jnp.argmax(policy))
        _, _, done = env.step(action)
    return rng, env.score


def evaluate_model(
    model: DropStackNet,
    params: Dict,
    *,
    games: int = 50,
    seed: int = 0,
    simulations: int = 20,
    c_puct: float = 1.0,
) -> float:
    """Return the average score over ``games`` greedy self-play episodes."""
    rng = jax.random.PRNGKey(seed)
    predict = jax.jit(model.apply)
    total = 0.0
    for _ in range(games):
        rng, key = jax.random.split(rng)
        key, score = play_game(
            model,
            params,
            key,
            simulations=simulations,
            c_puct=c_puct,
            predict=predict,
        )
        total += score
    return total / games
