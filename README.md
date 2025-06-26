# drop-stack-2048-ai

This repository provides a simple AlphaZero-style training pipeline for the
Drop Stack 2048 game. The new `main.py` script runs a full cycle consisting of
self-play data generation followed by network training.

The script now supports running multiple cycles in succession using the
``--cycles`` flag. Each cycle loads the latest checkpoint (if present) before
starting self-play so training can continue from the previously saved model.

Self-play uses Monte Carlo Tree Search to select actions. Early in each game the
agent samples moves in proportion to the MCTS visit counts. The `--greedy-after`
flag controls after how many moves the policy becomes greedy and always selects
the most-visited action.

Training uses an asynchronous data loader that prefetches batches on a
background thread so the GPU remains busy. Mixed precision can be enabled with
the `--mixed-precision` flag to reduce memory usage and speed up training.

You can speed up data generation by running multiple self-play episodes in
parallel using the `--processes` option.

After each training cycle the new model plays a set of test games to measure
its average score. If this score beats the previous best, the checkpoint is
promoted and stored alongside the current best score.

The training code sets a default value for the `JAX_TRACEBACK_FILTERING`
environment variable to disable traceback filtering. If you want different
behaviour, you can override this variable before running the training scripts.

## Installation

Install the required Python packages with::

    pip install -r requirements.txt

This installs `jax`, `flax`, `optax` and the build tools needed for the C++
extension.

For GPU acceleration you must install a CUDA-enabled `jaxlib` wheel. Follow the
[official JAX installation instructions](https://jax.readthedocs.io/en/latest/installation.html#pip-install)
and select the wheel that matches your CUDA and cuDNN versions. The `requirements.txt`
file pins the base packages, but you can replace the CPU build with::

    pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

If a GPU-capable `jaxlib` build is not installed, training will fall back to CPU
execution.

## Building the C++ extension

The merge logic is implemented in C++ and exposed to Python via `pybind11`.
After installing the dependencies, build the extension in-place with::

    python setup.py build_ext --inplace

This requires a C++17 compiler.

## CPU self-play workers

Parallel self-play uses multiprocessing to speed up data generation. Before
launching worker processes the `self_play_parallel` function clears
`CUDA_VISIBLE_DEVICES` so the workers run entirely on the CPU even when the
training step uses a GPU. JAX may report errors such as::

    Jax plugin configuration error: ... CUDA_ERROR_NO_DEVICE

These messages are harmless and simply indicate that the workers do not have GPU
access. You can redirect or suppress them if desired.

If you want the workers to evaluate the neural network on a GPU, pass
``use_gpu=True`` to ``self_play_parallel`` or the worker launch helpers. When
enabled, the MCTS rollouts keep running on the CPU but the network predictions
are offloaded to the GPU for better throughput.

## Asynchronous training

You can keep the GPU busy by generating self-play data while training. Pass the
`--workers` flag to `main.py` to launch background self-play processes that
continuously fill the replay buffer. The training loop fetches the latest
parameters for each batch so the workers automatically use the most recent
model.
