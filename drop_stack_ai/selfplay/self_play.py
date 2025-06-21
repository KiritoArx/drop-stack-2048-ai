from __future__ import annotations

from typing import List, Dict, Any, Tuple

import math

import jax
import jax.numpy as jnp

from drop_stack_ai.env.drop_stack_env import DropStackEnv
from drop_stack_ai.model.network import DropStackNet, create_model
from drop_stack_ai.model.mcts import run_mcts
from drop_stack_ai.training.replay_buffer import ReplayBuffer
from flax.serialization import to_bytes, from_bytes
import multiprocessing as mp


def _play_episode(
    model: DropStackNet,
    params: Dict[str, Any],
    rng: jax.random.PRNGKey,
    *,
    greedy: bool = False,
    greedy_after: int | None = None,
    simulations: int = 50,
    c_puct: float = 1.0,
) -> Tuple[jax.random.PRNGKey, List[Dict[str, Any]], List[jnp.ndarray], List[float], int]:
    """Return data for a single self-play episode."""
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
        values.append(0.0)

        _, _, done = env.step(action)
        step += 1

    final_score = math.log(env.score + 1)
    values = [float(final_score)] * len(values)
    return rng, states, policies, values, env.score


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
    """Run a single self-play episode using MCTS and store it in ``buffer``."""
    print("[self_play] starting episode")
    rng, states, policies, values, score = _play_episode(
        model,
        params,
        rng,
        greedy=greedy,
        greedy_after=greedy_after,
        simulations=simulations,
        c_puct=c_puct,
    )
    buffer.add_episode(states, policies, values)
    print(f"[self_play] finished episode with score={score}")
    return rng


def _worker(args: Tuple[int, bytes, int, bool, int | None, int, float]):
    """Helper for ``self_play_parallel`` running in a separate process."""
    seed, params_bytes, hidden_size, greedy, greedy_after, simulations, c_puct = args
    rng = jax.random.PRNGKey(seed)
    model, params = create_model(rng, hidden_size=hidden_size)
    params = from_bytes(params, params_bytes)
    _, states, policies, values, _ = _play_episode(
        model,
        params,
        rng,
        greedy=greedy,
        greedy_after=greedy_after,
        simulations=simulations,
        c_puct=c_puct,
    )
    return states, policies, values


def self_play_parallel(
    model: DropStackNet,
    params: Dict[str, Any],
    rng: jax.random.PRNGKey,
    buffer: ReplayBuffer,
    *,
    episodes: int,
    processes: int | None = None,
    greedy: bool = False,
    greedy_after: int | None = None,
    simulations: int = 50,
    c_puct: float = 1.0,
) -> jax.random.PRNGKey:
    """Run ``episodes`` self-play games in parallel."""
    params_bytes = to_bytes(params)
    keys = jax.random.split(rng, episodes + 1)
    rng = keys[0]
    seeds = [int(jax.random.randint(k, (), 0, 2**31 - 1)) for k in keys[1:]]

    args = [
        (
            seed,
            params_bytes,
            model.hidden_size,
            greedy,
            greedy_after,
            simulations,
            c_puct,
        )
        for seed in seeds
    ]

    with mp.Pool(processes) as pool:
        results = pool.map(_worker, args)

    for states, policies, values in results:
        buffer.add_episode(states, policies, values)

    return rng
