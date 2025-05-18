from collections import defaultdict, deque

class SocialNetwork:
    def __init__(self):
        self.graph = defaultdict(list)
        self.news_sources = set()

    def add_connection(self, person1, person2):
        self.graph[person1].append(person2)
        self.graph[person2].append(person1)

    def spread_news(self, source):
        visited = set()
        queue = deque()
        queue.append(source)
        visited.add(source)
        self.news_sources.add(source)

        print(f"News started spreading from: {source}")
        while queue:
            person = queue.popleft()
            for neighbor in self.graph[person]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    print(f"{person} told {neighbor}")
        
        return visited

    def detect_fake_news(self, infected_set):
        # If the news originates from multiple disconnected groups, it might be fake
        groups = 0
        visited = set()

        def dfs(node):
            visited.add(node)
            for neighbor in self.graph[node]:
                if neighbor in infected_set and neighbor not in visited:
                    dfs(neighbor)

        for person in infected_set:
            if person not in visited:
                dfs(person)
                groups += 1
        
        print(f"Infected groups detected: {groups}")
        return groups > 1

# Example usage
network = SocialNetwork()

# Add some connections
connections = [
    ('A', 'B'), ('B', 'C'), ('C', 'D'),
    ('E', 'F'), ('F', 'G'),
    ('H', 'I')  # Disconnected cluster
]

for p1, p2 in connections:
    network.add_connection(p1, p2)

# Spread news from multiple sources
infected1 = network.spread_news('A')
infected2 = network.spread_news('E')

# Union of infected people
total_infected = infected1.union(infected2)

# Detect if fake news
if network.detect_fake_news(total_infected):
    print("Warning: The news is likely fake.")
else:
    print("News seems genuine.")
