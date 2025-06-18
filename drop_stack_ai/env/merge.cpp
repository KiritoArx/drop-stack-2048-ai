#include "merge.hpp"
#include <deque>
#include <iomanip>
#include <iostream>
#include <string>

/* Pretty-printer (tight ASCII) */
void printBoard(const Board& board)
{
    for (int row = MAX_HEIGHT - 1; row >= 0; --row) {
        for (int col = 0; col < COLUMN_COUNT; ++col) {
            if (row < static_cast<int>(board[col].size()))
                std::cout << std::setw(6) << board[col][row];
            else
                std::cout << std::setw(6) << '.';
        }
        std::cout << '\n';
    }
    std::cout << std::string(COLUMN_COUNT * 6, '-') << '\n';
    for (int col = 0; col < COLUMN_COUNT; ++col)
        std::cout << std::setw(6) << ("C" + std::to_string(col));
    std::cout << "\n\n";
}

/* Drop-and-resolve (bug-free version) */
void dropAndResolve(Board& board, int value, int col)
{
    /* 1.  Drop the new tile */
    board[col].push_back(value);
    int row = static_cast<int>(board[col].size()) - 1;

    /* Remember where that brand-new tile landed */
    const int landingCol = col;
    const int landingRow = row;

    /* 2.  Breadth-first merge resolution */
    std::deque<std::pair<int, int>> fresh{ {col, row} };

    while (!fresh.empty()) {
        auto [c, r] = fresh.front();
        fresh.pop_front();

        if (r >= static_cast<int>(board[c].size())) continue;
        int v = board[c][r];

        /* vertical-down */
        if (r > 0 && board[c][r - 1] == v) {
            board[c][r - 1] *= 2;
            board[c].erase(board[c].begin() + r);
            fresh.push_front({ c, r - 1 });
            if (r < static_cast<int>(board[c].size()))
                fresh.push_back({ c, r });
            continue;
        }

        /* vertical-up */
        if (r + 1 < static_cast<int>(board[c].size()) &&
            board[c][r + 1] == v)
        {
            board[c][r] *= 2;
            board[c].erase(board[c].begin() + r + 1);
            fresh.push_front({ c, r });
            if (r + 1 < static_cast<int>(board[c].size()))
                fresh.push_back({ c, r + 1 });
            continue;
        }

        /* horizontal */
        for (int dc : {-1, 1}) {
            int nc = c + dc;
            if (nc < 0 || nc >= COLUMN_COUNT) continue;
            if (r < static_cast<int>(board[nc].size()) &&
                board[nc][r] == v)
            {
                /* --- only change starts here --- */
                bool keepRight = (r == landingRow && nc == landingCol);
                int  dst = keepRight ? nc : c;   // who keeps the merged tile
                int  src = keepRight ? c : nc;  // who gets erased

                board[dst][r] *= 2;                    // merge
                board[src].erase(board[src].begin() + r);
                /* --- only change ends here --- */

                fresh.push_front({ dst, r });
                if (r < static_cast<int>(board[nc].size()))
                    fresh.push_back({ src, r });
                break;
            }
        }
    }
}

bool gameOver(const Board& board)
{
    for (const auto& col : board)
        if (static_cast<int>(col.size()) >= MAX_HEIGHT)
            return true;
    return false;
}
