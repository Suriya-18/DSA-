#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int countChange(const vector<int>& denoms, int amount) {
    vector<int> dp(amount + 1, amount + 1);
    dp[0] = 0;
    for (int i = 1; i <= amount; ++i) {
        for (int coin : denoms) {
            if (i - coin >= 0)
                dp[i] = min(dp[i], dp[i - coin] + 1);
        }
    }
    return dp[amount] > amount ? -1 : dp[amount];
}

int main() {
    vector<int> denoms = {25, 10, 5, 1};
    int amount = 10;
    cout << countChange(denoms, amount) << endl;
    return 0;
}
