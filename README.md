# drop-stack-2048-ai

This repository provides a simple AlphaZero-style training pipeline for the
Drop Stack 2048 game. The new `main.py` script runs a full cycle consisting of
self-play data generation followed by network training.

The script now supports running multiple cycles in succession using the
``--cycles`` flag. Each cycle loads the latest checkpoint (if present) before
starting self-play so training can continue from the previously saved model.

After each training cycle the new model plays a set of test games to measure
its average score. If this score beats the previous best, the checkpoint is
promoted and stored alongside the current best score.

The training code sets a default value for the `JAX_TRACEBACK_FILTERING`
environment variable to disable traceback filtering. If you want different
behaviour, you can override this variable before running the training scripts.
