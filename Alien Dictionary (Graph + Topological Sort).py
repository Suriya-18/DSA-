#Given a sorted list of words in an unknown language, determine the order of characters
from collections import defaultdict, deque

def alienOrder(words):
    # Step 1: Build graph
    graph = defaultdict(set)
    in_degree = {char: 0 for word in words for char in word}

    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i+1]
        min_len = min(len(w1), len(w2))

        if w1[:min_len] == w2[:min_len] and len(w1) > len(w2):
            return ""  # Invalid ordering

        for c1, c2 in zip(w1, w2):
            if c1 != c2:
                if c2 not in graph[c1]:
                    graph[c1].add(c2)
                    in_degree[c2] += 1
                break

    # Step 2: Perform Topological Sort (BFS)
    queue = deque([char for char in in_degree if in_degree[char] == 0])
    order = []

    while queue:
        char = queue.popleft()
        order.append(char)
        for neighbor in graph[char]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return "".join(order) if len(order) == len(in_degree) else ""

# Example
words = ["wrt", "wrf", "er", "ett", "rftt"]
print(alienOrder(words))  # Output: "wertf"
