# Define memory for Experience Replay
from collections import deque
import random

import torch
import numpy as np

# from memory.tree import SumTree
from buffers.utils import device



# Simle Implementation of Experience Replay
# class ReplayMemory():
#     def __init__(self, maxlen, seed=None):
#         self.memory = deque([], maxlen=maxlen)

#         # Optional seed for reproducibility
#         if seed is not None:
#             random.seed(seed)

#     def append(self, transition):
#         self.memory.append(transition)

#     def sample(self, sample_size):
#         return random.sample(self.memory, sample_size)

#     def __len__(self):
#         return len(self.memory)



#  Advanced Implementation of Experience Replay

class ReplayBuffer:
    def __init__(self, state_size, action_size, buffer_size):
        # Transition buffers: state, action, reward, next_state, done
        self.state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.action = torch.empty(buffer_size, dtype=torch.long)  # Discrete actions
        self.reward = torch.empty(buffer_size, dtype=torch.float)
        self.next_state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.done = torch.empty(buffer_size, dtype=torch.bool)

        self.count = 0
        self.real_size = 0
        self.size = buffer_size

    def add(self, transition):
        state, action, reward, next_state, done = transition

        # Ensure correct data types and shapes
        self.state[self.count] = torch.as_tensor(state, dtype=torch.float)
        self.action[self.count] = int(action)  # Store scalar discrete action
        self.reward[self.count] = float(reward)
        self.next_state[self.count] = torch.as_tensor(next_state, dtype=torch.float)
        self.done[self.count] = bool(done)

        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def sample(self, batch_size):
        assert self.real_size >= batch_size, "Not enough samples in the buffer"

        sample_idxs = np.random.choice(self.real_size, batch_size, replace=False)

        states = self.state[sample_idxs].to(device())                      # [B, state_dim]
        actions = self.action[sample_idxs].to(device()).long().view(-1)   # [B]
        rewards = self.reward[sample_idxs].to(device()).view(-1)          # [B]
        next_states = self.next_state[sample_idxs].to(device())           # [B, state_dim]
        dones = self.done[sample_idxs].to(device()).float().view(-1)      # [B]

        return (states, actions, rewards, next_states, dones)

