import argparse
import os
import pickle
import time

# Force CPU-only usage regardless of system configuration
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import jax
import jax.numpy as jnp

from drop_stack_ai.model.network import create_model
from drop_stack_ai.selfplay.self_play import self_play_parallel
from drop_stack_ai.training.replay_buffer import ReplayBuffer
from drop_stack_ai.utils.serialization import load_params, save_bytes


def load_model(model_path: str, model, params):
    try:
        new_params = load_params(model_path, params)
        return new_params
    except FileNotFoundError:
        print(f"[actor] model {model_path} not found; using existing params")
        return params


DEFAULT_BUCKET = os.environ.get("DROPSTACK_BUCKET", "gs://drop-stack-ai-data-12345")
DEFAULT_MODEL = os.path.join(DEFAULT_BUCKET, "checkpoints", "model.msgpack")
DEFAULT_EPISODES = os.path.join(DEFAULT_BUCKET, "episodes")


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-play actor")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Path to model parameters",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_EPISODES,
        help="Directory or gs:// bucket for episodes",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5000,
        help="Episodes per self-play batch",
    )
    parser.add_argument(
        "--episodes-per-file",
        type=int,
        default=20000,
        help="Number of episodes per uploaded file",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=os.cpu_count() - 2,
        help="Parallel self-play processes (all cores minus two)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=1024,
        help="Model hidden size (match learner)",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Use float16 precision (only if learner uses fp16)",
    )
    parser.add_argument(
        "--greedy-after",
        type=int,
        default=15,
        help="Moves before greedy play (fine-tune exploration)",
    )
    parser.add_argument(
        "--sleep",
        type=int,
        default=0,
        help="Pause between batches (seconds)",
    )
    parser.add_argument(
        "--refresh-every",
        type=int,
        default=30,
        help="Seconds between checks for a new model",
    )
    parser.add_argument("--seed", type=int, default=42, help="PRNG seed")
    args = parser.parse_args()

    rng = jax.random.PRNGKey(args.seed)
    dtype = jnp.float16 if args.mixed_precision else jnp.float32
    model, params = create_model(rng, hidden_size=args.hidden_size, dtype=dtype)
    params = load_model(args.model, model, params)
    print(f"[actor] starting self-play with model {args.model}")
    last_stamp = None
    batch_buffer = ReplayBuffer()
    score_accum: list[float] = []
    step_accum: list[int] = []

    last_check = 0.0
    while True:
        now = time.time()
        if now - last_check >= args.refresh_every:
            last_check = now
            # Reload model if file changed
            try:
                if args.model.startswith("gs://"):
                    from google.cloud import storage

                    bucket, blob_name = args.model[5:].split("/", 1)
                    client = storage.Client()
                    blob = client.bucket(bucket).blob(blob_name)
                    blob.reload()
                    stamp = blob.updated
                else:
                    stamp = os.path.getmtime(args.model)
            except Exception:
                stamp = None
            if stamp != last_stamp:
                last_stamp = stamp
                params = load_model(args.model, model, params)
                print("[actor] loaded new model - starting on new model now")

        buffer = ReplayBuffer()
        rng, scores, steps = self_play_parallel(
            model,
            params,
            rng,
            buffer,
            episodes=args.episodes,
            processes=args.processes or os.cpu_count(),
            greedy_after=args.greedy_after,
            return_info=True,
        )
        print(f"[actor] completed {len(buffer.episodes)} episodes")
        score_accum.extend(scores)
        step_accum.extend(steps)
        batch_buffer.extend(buffer)
        if len(batch_buffer.episodes) >= args.episodes_per_file:
            payload = pickle.dumps(batch_buffer)
            filename = f"episodes_{int(time.time())}.pkl"
            out_path = os.path.join(args.output, filename)
            avg_score = sum(score_accum) / len(score_accum)
            avg_steps = sum(step_accum) / len(step_accum)
            print(
                f"[actor] uploading {len(batch_buffer.episodes)} episodes | Avg Score: {avg_score:.2f} | Avg Steps: {avg_steps:.2f}"
            )
            save_bytes(payload, out_path)
            print(f"[actor] uploaded to {out_path}")
            batch_buffer = ReplayBuffer()
            score_accum.clear()
            step_accum.clear()
        if args.sleep:
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
