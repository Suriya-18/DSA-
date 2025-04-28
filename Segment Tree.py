class SegmentTree:
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [0] * (2 * self.n)
        for i in range(self.n):
            self.tree[self.n + i] = arr[i]
        for i in range(self.n - 1, 0, -1):
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]

    def update(self, index, value):
        index += self.n
        self.tree[index] = value
        while index > 1:
            index //= 2
            self.tree[index] = self.tree[2 * index] + self.tree[2 * index + 1]

    def query(self, left, right):
        res = 0
        left += self.n
        right += self.n
        while left < right:
            if left % 2:
                res += self.tree[left]
                left += 1
            if right % 2:
                right -= 1
                res += self.tree[right]
            left //= 2
            right //= 2
        return res

# Example
arr = [2, 4, 5, 7, 8, 9]
st = SegmentTree(arr)
print(st.query(1, 4))  # Sum from index 1 to 3 => 4 + 5 + 7 = 16
