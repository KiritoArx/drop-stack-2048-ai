from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
from collections import deque


@dataclass
class ReplayBuffer:
    """Simple replay buffer storing recent episodes."""

    max_episodes: int = 200_000
    data: List[Dict[str, Any]] = field(default_factory=list)
    episodes: deque = field(init=False)

    def __post_init__(self) -> None:
        # ``episodes`` tracks the individual episodes so old ones can be
        # discarded once ``max_episodes`` is exceeded.
        self.episodes = deque()
        if self.data:
            # Assume entire data belongs to a single episode when loading from
            # legacy buffers.
            self.episodes.append(list(self.data))

    def add_episode(
        self, states: List[Dict[str, Any]], policies: List[Any], values: List[float]
    ) -> None:
        """Add an episode worth of data to the buffer."""
        episode = []
        for s, p, v in zip(states, policies, values):
            item = {"state": s, "policy": p, "value": v}
            episode.append(item)
            self.data.append(item)

        self.episodes.append(episode)
        while len(self.episodes) > self.max_episodes:
            old = self.episodes.popleft()
            del self.data[: len(old)]

    def __len__(self) -> int:
        return len(self.data)

    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """Randomly sample ``batch_size`` elements from the buffer."""
        import numpy as np

        n = len(self.data)
        if n == 0:
            return []
        replace = n < batch_size
        idx = np.random.choice(n, size=batch_size, replace=replace)
        return [self.data[i] for i in idx]

    def extend(self, other: "ReplayBuffer") -> None:
        """Append all episodes from ``other`` into this buffer."""
        for episode in other.episodes:
            states = [item["state"] for item in episode]
            policies = [item["policy"] for item in episode]
            values = [item["value"] for item in episode]
            self.add_episode(states, policies, values)
