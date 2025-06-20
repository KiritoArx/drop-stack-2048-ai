# drop-stack-2048-ai

This repository provides a simple AlphaZero-style training pipeline for the
Drop Stack 2048 game. The new `main.py` script runs a full cycle consisting of
self-play data generation followed by network training.

The training code sets a default value for the `JAX_TRACEBACK_FILTERING`
environment variable to disable traceback filtering. If you want different
behaviour, you can override this variable before running the training scripts.
