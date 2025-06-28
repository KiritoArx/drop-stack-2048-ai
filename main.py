import os
import argparse

os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")

import jax
import jax.numpy as jnp

from drop_stack_ai.utils.device_info import print_device_info

from drop_stack_ai.model.network import create_model
from drop_stack_ai.selfplay.self_play import self_play, self_play_parallel
from drop_stack_ai.training.replay_buffer import ReplayBuffer
from drop_stack_ai.training.train import TrainConfig, train
from drop_stack_ai.utils.serialization import load_params
from drop_stack_ai.selfplay.evaluate import evaluate_model


def run_cycle(
    episodes: int,
    seed: int,
    config: TrainConfig,
    *,
    greedy_after: int | None = None,
    processes: int | None = None,
) -> None:
    """Run self-play to populate a buffer then train a model."""
    print(
        f"[run_cycle] starting: episodes={episodes} seed={seed} hidden_size={config.hidden_size}"
    )
    rng = jax.random.PRNGKey(seed)
    dtype = jnp.float16 if config.mixed_precision else jnp.float32
    model, params = create_model(rng, hidden_size=config.hidden_size, dtype=dtype)
    if config.checkpoint_path and os.path.exists(config.checkpoint_path):
        print(f"[run_cycle] loading checkpoint from {config.checkpoint_path}")
        params = load_params(config.checkpoint_path, params)
    buffer = ReplayBuffer(max_episodes=config.buffer_size)
    if config.workers == 0:
        if episodes > 0:
            if processes is None or processes > 1:
                rng = self_play_parallel(
                    model,
                    params,
                    rng,
                    buffer,
                    episodes=episodes,
                    processes=processes,
                    greedy_after=greedy_after,
                )
                print(
                    f"[run_cycle] buffer size={len(buffer)} after parallel self-play"
                )
            else:
                for i in range(episodes):
                    print(f"[run_cycle] self-play episode {i + 1}/{episodes}")
                    rng = self_play(
                        model, params, rng, buffer, greedy_after=greedy_after
                    )
                    print(
                        f"[run_cycle] buffer size={len(buffer)} after episode {i + 1}"
                    )
        if len(buffer) == 0:
            raise ValueError("Replay buffer is empty after self-play")
    else:
        print(
            f"[run_cycle] asynchronous mode with {config.workers} workers; episodes arg ignored"
        )
    print("[run_cycle] starting training")
    train(buffer, seed=seed, config=config)
    print("[run_cycle] training complete")

    # Evaluate the newly trained model
    if config.checkpoint_path:
        print("[run_cycle] evaluating model")
        params = load_params(config.checkpoint_path, params)
        avg_score = evaluate_model(model, params, games=50, seed=seed)
        best_path = config.checkpoint_path + ".best"
        score_path = best_path + ".txt"
        best_score = 0.0
        if os.path.exists(score_path):
            with open(score_path) as f:
                best_score = float(f.read())
        print(f"[run_cycle] average score={avg_score:.2f} best={best_score:.2f}")
        if avg_score > best_score:
            import shutil

            shutil.copy(config.checkpoint_path, best_path)
            with open(score_path, "w") as f:
                f.write(str(avg_score))
            print("[run_cycle] promoted new model")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Drop Stack 2048 training cycle")
    parser.add_argument("--episodes", type=int, default=20000, help="Number of self-play episodes")
    parser.add_argument("--steps", type=int, default=100000, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--hidden-size", type=int, default=1024, help="Model hidden size")
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable mixed precision training",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join("checkpoints", "model.msgpack"),
        help="Path to save or load model parameters",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--cycles", type=int, default=20, help="Number of training cycles to run")
    parser.add_argument(
        "--greedy-after",
        type=int,
        default=10,
        help="Number of moves to sample probabilistically before switching to greedy play",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help="Number of worker processes for self-play (default: all CPUs)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Background self-play workers during training",
    )
    args = parser.parse_args()

    os.environ.setdefault("JAX_TRACEBACK_FILTERING", "off")

    print_device_info()

    config = TrainConfig(
        batch_size=args.batch_size,
        steps=args.steps,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        checkpoint_path=args.checkpoint,
        mixed_precision=args.mixed_precision,
        workers=args.workers,
    )
    for cycle in range(args.cycles):
        print(f"[main] starting cycle {cycle + 1}/{args.cycles}")
        run_cycle(
            args.episodes,
            args.seed + cycle,
            config,
            greedy_after=args.greedy_after,
            processes=args.processes,
        )


if __name__ == "__main__":
    main()
