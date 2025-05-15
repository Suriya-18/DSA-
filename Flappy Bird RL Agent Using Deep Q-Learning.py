import pygame
import random
import numpy as np

pygame.init()

# Constants
WIDTH, HEIGHT = 500, 700
GRAVITY = 0.5
JUMP_VELOCITY = -10
PIPE_GAP = 200
PIPE_WIDTH = 80

class Bird:
    def __init__(self):
        self.y = HEIGHT // 2
        self.x = 100
        self.velocity = 0
        self.score = 0

    def jump(self):
        self.velocity = JUMP_VELOCITY

    def move(self):
        self.velocity += GRAVITY
        self.y += self.velocity

    def get_rect(self):
        return pygame.Rect(self.x, self.y, 30, 30)

class Pipe:
    def __init__(self, x):
        self.x = x
        self.height = random.randint(100, HEIGHT - PIPE_GAP - 100)

    def move(self):
        self.x -= 5

    def get_rects(self):
        top = pygame.Rect(self.x, 0, PIPE_WIDTH, self.height)
        bottom = pygame.Rect(self.x, self.height + PIPE_GAP, PIPE_WIDTH, HEIGHT)
        return top, bottom

class FlappyEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.bird = Bird()
        self.pipes = [Pipe(600)]
        self.done = False
        self.score = 0
        return self.get_state()

    def step(self, action):
        if action == 1:
            self.bird.jump()
        self.bird.move()
        for pipe in self.pipes:
            pipe.move()

        if self.pipes[-1].x < 300:
            self.pipes.append(Pipe(600))

        if self.pipes[0].x < -PIPE_WIDTH:
            self.pipes.pop(0)
            self.score += 1

        reward = 0.1
        bird_rect = self.bird.get_rect()
        for pipe in self.pipes:
            top, bottom = pipe.get_rects()
            if bird_rect.colliderect(top) or bird_rect.colliderect(bottom):
                self.done = True
                reward = -10

        if self.bird.y > HEIGHT or self.bird.y < 0:
            self.done = True
            reward = -10

        return self.get_state(), reward, self.done

    def get_state(self):
        pipe = self.pipes[0]
        return np.array([
            self.bird.y / HEIGHT,
            pipe.x / WIDTH,
            pipe.height / HEIGHT,
            self.bird.velocity / 10
        ], dtype=np.float32)


#DQN Agent Using TensorFlow
import tensorflow as tf
from collections import deque
import random

model = tf.keras.Sequential([
    tf.keras.layers.Dense(24, input_shape=(4,), activation='relu'),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(2, activation='linear')
])
model.compile(optimizer='adam', loss='mse')

memory = deque(maxlen=2000)
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
batch_size = 64

def act(state):
    if np.random.rand() < epsilon:
        return random.randint(0, 1)
    q_values = model.predict(state[np.newaxis], verbose=0)
    return np.argmax(q_values[0])

def train_model():
    if len(memory) < batch_size:
        return
    minibatch = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            target += gamma * np.amax(model.predict(next_state[np.newaxis], verbose=0)[0])
        target_f = model.predict(state[np.newaxis], verbose=0)
        target_f[0][action] = target
        model.fit(state[np.newaxis], target_f, epochs=1, verbose=0)


#Training Loop

env = FlappyEnv()
episodes = 1000

for ep in range(episodes):
    state = env.reset()
    total_reward = 0

    while True:
        action = act(state)
        next_state, reward, done = env.step(action)
        memory.append((state, action, reward, next_state, done))
        train_model()
        state = next_state
        total_reward += reward

        if done:
            break

    global epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    print(f"Episode {ep+1}: Score={env.score} | Total Reward={total_reward:.2f}")
