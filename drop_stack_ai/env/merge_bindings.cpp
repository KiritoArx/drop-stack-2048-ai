#include "merge.hpp"
#include <deque>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

int dropResolveAndScore(Board& board, int value, int col) {
    board[col].push_back(value);
    int row = static_cast<int>(board[col].size()) - 1;

    const int landingCol = col;
    const int landingRow = row;

    int reward = 0;
    std::deque<std::pair<int, int>> fresh{{col, row}};

    while (!fresh.empty()) {
        auto [c, r] = fresh.front();
        fresh.pop_front();

        if (r >= static_cast<int>(board[c].size())) continue;
        int v = board[c][r];

        if (r > 0 && board[c][r - 1] == v) {
            board[c][r - 1] *= 2;
            reward += board[c][r - 1];
            board[c].erase(board[c].begin() + r);
            fresh.push_front({c, r - 1});
            if (r < static_cast<int>(board[c].size()))
                fresh.push_back({c, r});
            continue;
        }

        if (r + 1 < static_cast<int>(board[c].size()) && board[c][r + 1] == v) {
            board[c][r] *= 2;
            reward += board[c][r];
            board[c].erase(board[c].begin() + r + 1);
            fresh.push_front({c, r});
            if (r + 1 < static_cast<int>(board[c].size()))
                fresh.push_back({c, r + 1});
            continue;
        }

        for (int dc : {-1, 1}) {
            int nc = c + dc;
            if (nc < 0 || nc >= static_cast<int>(board.size())) continue;
            if (r < static_cast<int>(board[nc].size()) && board[nc][r] == v) {
                bool keepRight = (r == landingRow && nc == landingCol);
                int dst = keepRight ? nc : c;
                int src = keepRight ? c : nc;

                board[dst][r] *= 2;
                reward += board[dst][r];
                board[src].erase(board[src].begin() + r);

                fresh.push_front({dst, r});
                if (r < static_cast<int>(board[nc].size()))
                    fresh.push_back({src, r});
                break;
            }
        }
    }

    return reward;
}

PYBIND11_MODULE(_merge_cpp, m) {
    m.doc() = "pybind11 wrapper for merge.cpp";

    m.def("drop_resolve_and_score", [](Board board, int value, int col) {
        int reward = dropResolveAndScore(board, value, col);
        return std::make_pair(board, reward);
    });

    m.def("drop_and_resolve", [](Board board, int value, int col) {
        dropAndResolve(board, value, col);
        return board;
    });

    m.def("game_over", [](const Board& board, int max_height) {
        for (const auto& column : board)
            if (static_cast<int>(column.size()) >= max_height)
                return true;
        return false;
    });

    m.def("print_board", &printBoard);
}
