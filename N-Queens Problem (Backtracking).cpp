#include <iostream>
#include <vector>
using namespace std;

void solve(int row, vector<string>& board, vector<vector<string>>& res, int n,
           vector<bool>& cols, vector<bool>& d1, vector<bool>& d2) {
    if (row == n) {
        res.push_back(board);
        return;
    }

    for (int col = 0; col < n; ++col) {
        if (cols[col] || d1[row + col] || d2[row - col + n - 1]) continue;

        board[row][col] = 'Q';
        cols[col] = d1[row + col] = d2[row - col + n - 1] = true;

        solve(row + 1, board, res, n, cols, d1, d2);

        board[row][col] = '.';
        cols[col] = d1[row + col] = d2[row - col + n - 1] = false;
    }
}

vector<vector<string>> solveNQueens(int n) {
    vector<vector<string>> res;
    vector<string> board(n, string(n, '.'));
    vector<bool> cols(n, false), d1(2 * n - 1, false), d2(2 * n - 1, false);
    solve(0, board, res, n, cols, d1, d2);
    return res;
}

int main() {
    int n = 4;
    vector<vector<string>> res = solveNQueens(n);
    for (auto& solution : res) {
        for (auto& row : solution) {
            cout << row << endl;
        }
        cout << "----" << endl;
    }
}
