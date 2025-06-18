"""Python implementation of the merge logic for Drop Stack 2048.

This mirrors the behaviour of the C++ code in ``merge.cpp``.  A board is a
list of columns, each column being a list of tile values with the bottom
entry at index ``0``.
"""
from collections import deque
from typing import List


Board = List[List[int]]


def print_board(board: Board, max_height: int | None = None) -> None:
    """Pretty print the board using a tight ASCII layout."""
    column_count = len(board)
    if max_height is None:
        max_height = max((len(col) for col in board), default=0)

    for row in range(max_height - 1, -1, -1):
        line = []
        for col in range(column_count):
            if row < len(board[col]):
                line.append(f"{board[col][row]:6d}")
            else:
                line.append("      .")
        print("".join(line))
    print("-" * (column_count * 6))
    print("".join(f"{f'C{c}':6}" for c in range(column_count)))
    print()


def drop_and_resolve(board: Board, value: int, col: int) -> None:
    """Drop a tile into ``col`` and resolve all resulting merges."""
    # 1. Drop the new tile
    board[col].append(value)
    row = len(board[col]) - 1

    landing_col = col
    landing_row = row

    # 2. Breadth-first merge resolution
    fresh: deque[tuple[int, int]] = deque([(col, row)])

    while fresh:
        c, r = fresh.popleft()

        if r >= len(board[c]):
            continue
        v = board[c][r]

        # vertical-down
        if r > 0 and board[c][r - 1] == v:
            board[c][r - 1] *= 2
            del board[c][r]
            fresh.appendleft((c, r - 1))
            if r < len(board[c]):
                fresh.append((c, r))
            continue

        # vertical-up
        if r + 1 < len(board[c]) and board[c][r + 1] == v:
            board[c][r] *= 2
            del board[c][r + 1]
            fresh.appendleft((c, r))
            if r + 1 < len(board[c]):
                fresh.append((c, r + 1))
            continue

        # horizontal
        for dc in (-1, 1):
            nc = c + dc
            if nc < 0 or nc >= len(board):
                continue
            if r < len(board[nc]) and board[nc][r] == v:
                keep_right = r == landing_row and nc == landing_col
                dst = nc if keep_right else c
                src = c if keep_right else nc

                board[dst][r] *= 2
                del board[src][r]

                fresh.appendleft((dst, r))
                if r < len(board[nc]):
                    fresh.append((src, r))
                break


def game_over(board: Board, max_height: int) -> bool:
    """Return ``True`` if any column reached ``max_height``."""
    return any(len(col) >= max_height for col in board)
