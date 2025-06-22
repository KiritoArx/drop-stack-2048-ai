from __future__ import annotations

__all__ = ["data_loader"]

import threading
import time
from queue import Queue
from typing import Iterator, Dict, Any

import jax
import jax.numpy as jnp

from .replay_buffer import ReplayBuffer
from drop_stack_ai.utils.state_utils import state_to_arrays


def _prepare_batch(samples: list[Dict[str, Any]]) -> Dict[str, jnp.ndarray]:
    boards = []
    currents = []
    nexts = []
    policies = []
    values = []
    for item in samples:
        b, c, n = state_to_arrays(item["state"])
        boards.append(b)
        currents.append(c)
        nexts.append(n)
        policy = jnp.array(item["policy"], dtype=jnp.float32)
        if policy.ndim == 1 and policy.sum() != 0:
            policy = policy / policy.sum()
        policies.append(policy)
        values.append(jnp.array(item["value"], dtype=jnp.float32))
    return {
        "board": jnp.stack(boards),
        "current": jnp.stack(currents),
        "next": jnp.stack(nexts),
        "policy": jnp.stack(policies),
        "value": jnp.stack(values),
    }


def _prefetch_generator(gen: Iterator[Dict[str, jnp.ndarray]], size: int) -> Iterator[Dict[str, jnp.ndarray]]:
    queue: Queue = Queue(maxsize=size)
    stop = object()

    def _loop() -> None:
        for item in gen:
            queue.put(item)
        queue.put(stop)

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()

    while True:
        item = queue.get()
        if item is stop:
            return
        yield item


def data_loader(
    buffer: ReplayBuffer,
    batch_size: int,
    *,
    devices: list[jax.Device],
    prefetch: int = 2,
) -> Iterator[Dict[str, jnp.ndarray]]:
    n_devices = len(devices)

    def _iterator() -> Iterator[Dict[str, jnp.ndarray]]:
        while True:
            while len(buffer) < batch_size:
                time.sleep(0.01)
            samples = buffer.sample(batch_size)
            batch = _prepare_batch(samples)
            if n_devices > 1:
                per_dev = batch_size // n_devices
                batch = {
                    k: v.reshape((n_devices, per_dev) + v.shape[1:])
                    for k, v in batch.items()
                }
                batch = {
                    k: jax.device_put_sharded(list(v), devices)
                    for k, v in batch.items()
                }
            else:
                batch = jax.device_put(batch, devices[0])
            yield batch

    return _prefetch_generator(_iterator(), prefetch)
