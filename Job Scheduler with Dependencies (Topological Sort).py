from collections import defaultdict, deque

def job_scheduler(jobs, deps):
    graph = defaultdict(list)
    indegree = {job: 0 for job in jobs}

    for a, b in deps:
        graph[a].append(b)
        indegree[b] += 1

    q = deque([job for job in jobs if indegree[job] == 0])
    result = []

    while q:
        job = q.popleft()
        result.append(job)
        for neighbor in graph[job]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                q.append(neighbor)

    return result if len(result) == len(jobs) else []

# Example
jobs = ['a', 'b', 'c', 'd']
deps = [('a', 'b'), ('b', 'c'), ('a', 'c'), ('c', 'd')]
print(job_scheduler(jobs, deps))  # ['a', 'b', 'c', 'd']
