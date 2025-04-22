import numpy as np
import matplotlib.pyplot as plt
import random

# Grid dimensions
ROWS, COLS = 7, 10
START = (3, 0)
GOAL = (3, 7)

# Wind strength for each column
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

# Actions: up, down, right, left
ACTIONS = [(-1, 0), (1, 0), (0, 1), (0, -1)]
ACTION_NAMES = ['U', 'D', 'R', 'L']

# Parameters
EPSILON = 0.1 # Exploration rate
ALPHA = 0.5 # Learning rate
GAMMA = 1.0 # Discount factor
EPISODES = 170

def step(state, action):
    row, col = state
    dr, dc = ACTIONS[action]
    new_row = row + dr - WIND[col]  # Wind shifts up
    new_col = col + dc
    new_row = min(max(new_row, 0), ROWS - 1)
    new_col = min(max(new_col, 0), COLS - 1)
    return (new_row, new_col)

def epsilon_greedy(Q, state):
    if np.random.rand() < EPSILON:
        return np.random.choice(len(ACTIONS))
    else:
        return np.argmax(Q[state])

def sarsa():
    Q = {}
    for row in range(ROWS):
        for col in range(COLS):
            Q[(row, col)] = np.zeros(len(ACTIONS))

    steps_per_episode = []

    time = 0
    while time < 8000:
        state = START
        action = epsilon_greedy(Q, state)
        steps = 0

        while state != GOAL:
            next_state = step(state, action)
            next_action = epsilon_greedy(Q, next_state)

            reward = -1
            Q[state][action] += ALPHA * (
                reward + GAMMA * Q[next_state][next_action] - Q[state][action]
            )

            state = next_state
            action = next_action
            steps += 1
            time += 1

        steps_per_episode.append((time, steps))

    return Q, steps_per_episode
episode_length=[16,18]
# Run the SARSA algorithm
Q, episode_data = sarsa()
# Plot results
times, steps = zip(*episode_data)
plt.plot(times, steps)
plt.xlabel('Time steps')
plt.ylabel('Episode length')
plt.title('SARSA on Windy Gridworld')
plt.grid(True)
plt.show()
# Extract episode lengths
times, steps = zip(*episode_data)
episode_lengths = [s for _, s in episode_data]
average_length = np.mean(episode_length)
optimal_steps = 15

# Plot episode lengths
plt.figure(figsize=(10, 6))
plt.plot(times, steps, label='Actual Episode Length', color='blue', alpha=0.7)

# Plot optimal path length
plt.axhline(y=optimal_steps, color='red', linestyle='--', label='Optimal Path (15 steps)')

# Plot average episode length
plt.axhline(y=average_length, color='green', linestyle='--', label=f'Avg Length ≈ 17 steps')

# Annotate the difference
plt.text(times[-1]*0.6, optimal_steps + 0.5, 'Optimal Path', color='red')
plt.text(times[-1]*0.4, average_length + 0.5, f'Average with ε-greedy', color='green')

# Labels and title
plt.xlabel('Time steps')
plt.ylabel('Episode length')
plt.title('SARSA on Windy Gridworld: Optimal vs Actual Episode Lengths')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
