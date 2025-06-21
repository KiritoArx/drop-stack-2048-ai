import os
import argparse

import jax

from drop_stack_ai.model.network import create_model
from drop_stack_ai.selfplay.self_play import self_play
from drop_stack_ai.training.replay_buffer import ReplayBuffer
from drop_stack_ai.training.train import TrainConfig, train


def run_cycle(episodes: int, seed: int, config: TrainConfig) -> None:
    """Run self-play to populate a buffer then train a model."""
    print(
        f"[run_cycle] starting: episodes={episodes} seed={seed} hidden_size={config.hidden_size}"
    )
    rng = jax.random.PRNGKey(seed)
    model, params = create_model(rng, hidden_size=config.hidden_size)
    buffer = ReplayBuffer()
    for i in range(episodes):
        print(f"[run_cycle] self-play episode {i + 1}/{episodes}")
        rng = self_play(model, params, rng, buffer)
        print(f"[run_cycle] buffer size={len(buffer)} after episode {i + 1}")
    if len(buffer) == 0:
        raise ValueError("Replay buffer is empty after self-play")
    print("[run_cycle] starting training")
    train(buffer, seed=seed, config=config)
    print("[run_cycle] training complete")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Drop Stack 2048 training cycle")
    parser.add_argument("--episodes", type=int, default=10, help="Number of self-play episodes")
    parser.add_argument("--steps", type=int, default=1000, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-size", type=int, default=128, help="Model hidden size")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    os.environ.setdefault("JAX_TRACEBACK_FILTERING", "off")

    config = TrainConfig(
        batch_size=args.batch_size,
        steps=args.steps,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
    )
    run_cycle(args.episodes, args.seed, config)


if __name__ == "__main__":
    main()
