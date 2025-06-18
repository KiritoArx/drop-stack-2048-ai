from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ReplayBuffer:
    """Simple in-memory replay buffer."""

    data: List[Dict[str, Any]] = field(default_factory=list)

    def add_episode(
        self, states: List[Dict[str, Any]], policies: List[Any], values: List[float]
    ) -> None:
        """Add an episode worth of data to the buffer."""
        for s, p, v in zip(states, policies, values):
            self.data.append({"state": s, "policy": p, "value": v})

    def __len__(self) -> int:
        return len(self.data)

    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """Randomly sample ``batch_size`` elements from the buffer."""
        import random

        return random.sample(self.data, batch_size)
