import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from sklearn.ensemble import RandomForestClassifier, IsolationForest

# DQN Neural Network Model
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

# Replay Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Environment Simulator
class SimpleEnv:
    def __init__(self, states, reward_fn):
        self.states = states
        self.reward_fn = reward_fn
        self.n = len(states)
        self.idx = 0

    def reset(self):
        self.idx = 0
        return self.states[self.idx]

    def step(self, action):
        state = self.states[self.idx]
        reward = self.reward_fn(state, action)
        self.idx += 1
        done = self.idx >= self.n
        next_state = self.states[self.idx] if not done else None
        return next_state, reward, done

def create_random_forest_classifier():
    return RandomForestClassifier(n_estimators=100, random_state=42)

def create_isolation_forest(contamination=0.05):
    return IsolationForest(n_estimators=100, contamination=contamination, random_state=42)