import numpy as np
import gym
import random

# Create the environment
env = gym.make('FrozenLake-v1', is_slippery=False)

# Initialize the Q-table with zeros
state_space_size = env.observation_space.n  # 16 states in FrozenLake
action_space_size = env.action_space.n  # 4 possible actions
Q_table = np.zeros((state_space_size, action_space_size))

# Parameters
learning_rate = 0.1
discount_factor = 0.99
epsilon = 0.1  # Exploration-exploitation trade-off
episodes = 1000

# Training loop
for episode in range(episodes):
    state = env.reset()  # Reset the environment at the start of each episode
    done = False
    total_reward = 0

    while not done:
        # Exploration vs. exploitation
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Exploration: choose a random action
        else:
            action = np.argmax(Q_table[state])  # Exploitation: choose the best action

        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward

        # Update Q-value using the Q-learning formula
        Q_table[state, action] = Q_table[state, action] + learning_rate * (
            reward + discount_factor * np.max(Q_table[next_state]) - Q_table[state, action]
        )

        state = next_state

    # Optional: print episode result
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")

# After training, we can test the learned policy
state = env.reset()
done = False
total_reward = 0

while not done:
    action = np.argmax(Q_table[state])  # Choose the best action from Q-table
    state, reward, done, _, _ = env.step(action)
    total_reward += reward

print("Total reward in test episode:", total_reward)
