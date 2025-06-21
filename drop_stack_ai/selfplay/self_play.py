from __future__ import annotations

from typing import List, Dict, Any

import math

import jax
import jax.numpy as jnp

from drop_stack_ai.env.drop_stack_env import DropStackEnv
from drop_stack_ai.model.network import DropStackNet
from drop_stack_ai.model.mcts import run_mcts
from drop_stack_ai.training.replay_buffer import ReplayBuffer


def self_play(
    model: DropStackNet,
    params: Dict[str, Any],
    rng: jax.random.PRNGKey,
    buffer: ReplayBuffer,
    *,
    greedy: bool = False,
    simulations: int = 50,
    c_puct: float = 1.0,
) -> jax.random.PRNGKey:
    """Run a single self-play episode using MCTS.

    At every step an MCTS search is performed starting from the current
    environment state. The resulting visit counts are normalised and stored as
    the policy target. Moves are selected proportionally to this distribution
    when ``greedy`` is ``False`` and greedily otherwise.
    """
    print("[self_play] starting episode")
    env = DropStackEnv()
    states: List[Dict[str, Any]] = []
    policies: List[jnp.ndarray] = []
    values: List[float] = []

    done = False
    while not done:
        raw_state = env.get_state()
        policy = run_mcts(
            model,
            params,
            env,
            num_simulations=simulations,
            c_puct=c_puct,
        )

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
    print(f"[self_play] finished episode with score={env.score}")
    return rng
