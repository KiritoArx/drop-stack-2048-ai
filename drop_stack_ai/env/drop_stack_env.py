"""Environment for the Drop Stack 2048 game."""

from __future__ import annotations

import random
from collections import deque
from copy import deepcopy
from typing import Dict, List

from .merge import drop_and_resolve, print_board, game_over, Board


class DropStackEnv:
    """Simple Drop Stack 2048 environment."""

    COLUMN_COUNT = 5
    MAX_HEIGHT = 6

    def __init__(self, *, seed: int | None = None) -> None:
        self.random = random.Random(seed)
        self.reset()

    # Internal helpers -----------------------------------------------------
    def _spawn_tile(self) -> int:
        """Spawn a new tile according to progressive unlock rules."""
        # Determine largest value we are allowed to spawn.
        max_spawn = max(2, self.max_tile // 2)
        possible: List[int] = [2]
        value = 4
        while value <= max_spawn:
            possible.append(value)
            value *= 2

        if len(possible) == 1:
            return 2

        # 2 is more common than the rest; distribute remaining probability
        base_prob = 0.75
        other_prob = (1.0 - base_prob) / (len(possible) - 1)
        weights = [base_prob] + [other_prob] * (len(possible) - 1)
        return self.random.choices(possible, weights=weights)[0]

    def _drop_resolve_and_score(self, board: Board, value: int, col: int) -> int:
        """Drop a tile on ``board`` and resolve merges, returning the score gain."""
        board[col].append(value)
        row = len(board[col]) - 1
        landing_col = col
        landing_row = row

        reward = 0
        fresh: deque[tuple[int, int]] = deque([(col, row)])

        while fresh:
            c, r = fresh.popleft()
            if r >= len(board[c]):
                continue
            v = board[c][r]

            # vertical-down
            if r > 0 and board[c][r - 1] == v:
                board[c][r - 1] *= 2
                reward += board[c][r - 1]
                del board[c][r]
                fresh.appendleft((c, r - 1))
                if r < len(board[c]):
                    fresh.append((c, r))
                continue

            # vertical-up
            if r + 1 < len(board[c]) and board[c][r + 1] == v:
                board[c][r] *= 2
                reward += board[c][r]
                del board[c][r + 1]
                fresh.appendleft((c, r))
                if r + 1 < len(board[c]):
                    fresh.append((c, r + 1))
                continue

            # horizontal
            for dc in (-1, 1):
                nc = c + dc
                if nc < 0 or nc >= self.COLUMN_COUNT:
                    continue
                if r < len(board[nc]) and board[nc][r] == v:
                    keep_right = r == landing_row and nc == landing_col
                    dst = nc if keep_right else c
                    src = c if keep_right else nc

                    board[dst][r] *= 2
                    reward += board[dst][r]
                    del board[src][r]

                    fresh.appendleft((dst, r))
                    if r < len(board[nc]):
                        fresh.append((src, r))
                    break
        return reward

    # Public API -----------------------------------------------------------
    def reset(self) -> Dict[str, object]:
        """Reset the environment to a new game."""
        self.board: Board = [[] for _ in range(self.COLUMN_COUNT)]
        self.score = 0
        self.max_tile = 2
        self.current_tile = 2
        self.next_tile = 2
        self.done = False

        # Spawn first two tiles
        self.current_tile = self._spawn_tile()
        self.next_tile = self._spawn_tile()
        return self.get_state()

    def step(self, action_col: int) -> tuple[Dict[str, object], int, bool]:
        """Drop the current tile into ``action_col``.

        Returns a tuple of ``(next_state, reward, done)``.
        """
        if self.done:
            return self.get_state(), 0, True

        if action_col < 0 or action_col >= self.COLUMN_COUNT:
            raise ValueError("Invalid column")

        sim_board = deepcopy(self.board)
        reward = self._drop_resolve_and_score(sim_board, self.current_tile, action_col)
        self.score += reward

        # Apply the actual drop using the reference implementation
        drop_and_resolve(self.board, self.current_tile, action_col)

        # Update max tile
        if self.board:
            self.max_tile = max(self.max_tile, max((max(col) for col in self.board if col), default=self.max_tile))

        # Advance tiles
        self.current_tile = self.next_tile
        self.next_tile = self._spawn_tile()

        # Check for game over
        self.done = game_over(self.board, self.MAX_HEIGHT)

        next_state = self.get_state()
        return next_state, reward, self.done

    def render(self) -> None:
        """Print the current board."""
        print_board(self.board, self.MAX_HEIGHT)
        print(f"Score: {self.score}\nCurrent: {self.current_tile}  Next: {self.next_tile}\n")

    def get_state(self) -> Dict[str, object]:
        """Return the current state of the game."""
        return {
            "board": deepcopy(self.board),
            "current_tile": self.current_tile,
            "next_tile": self.next_tile,
            "score": self.score,
            "done": self.done,
        }

