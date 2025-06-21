#pragma once
#include <vector>

constexpr int COLUMN_COUNT = 5;
constexpr int MAX_HEIGHT = 6;

using Board = std::vector<std::vector<int>>;

void printBoard(const Board& board);
void dropAndResolve(Board& board, int value, int col);
bool gameOver(const Board& board);
