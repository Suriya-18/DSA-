from collections import deque

class StackUsingQueue:
    def __init__(self):
        self.q = deque()
    
    def push(self, x):
        self.q.append(x)
        for _ in range(len(self.q) - 1):
            self.q.append(self.q.popleft())
    
    def pop(self):
        return self.q.popleft()
    
    def top(self):
        return self.q[0]
    
    def is_empty(self):
        return len(self.q) == 0

s = StackUsingQueue()
s.push(1)
s.push(2)
s.push(3)
print(s.pop())  # Output: 3
