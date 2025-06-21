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
    greedy_after: int | None = None,
    simulations: int = 50,
    c_puct: float = 1.0,
) -> jax.random.PRNGKey:
    """Play one episode and store the experience in ``buffer``.

    The first ``greedy_after`` moves are sampled from the MCTS policy
    distribution; subsequent moves take the greedy action. If ``greedy`` is
    ``True`` the episode is played greedily from the start.
    """
    env = DropStackEnv()
    states: List[Dict[str, Any]] = []
    policies: List[jnp.ndarray] = []
    values: List[float] = []

    step = 0
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

        use_greedy = greedy or (greedy_after is not None and step >= greedy_after)
        if use_greedy:
            action = int(jnp.argmax(policy))
        else:
            rng, key = jax.random.split(rng)
            action = int(jax.random.choice(key, 5, p=policy))

        states.append(raw_state)
        policies.append(policy)
        values.append(0.0)  # placeholder

        _, _, done = env.step(action)
        step += 1

    # Episode finished, assign final score as the value target
    final_score = math.log(env.score + 1)
    values = [float(final_score)] * len(values)
    buffer.add_episode(states, policies, values)
    return rng
