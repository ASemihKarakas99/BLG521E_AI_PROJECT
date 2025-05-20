import torch
import numpy as np


class AttentiveReplayBuffer:
    def __init__(self, state_size, action_size, buffer_size, lambda_factor=2):
        self.lambda_factor = lambda_factor
        self.state_size = state_size
        self.action_size = action_size
        self.size = buffer_size

        self.state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.action = torch.empty(buffer_size, action_size, dtype=torch.float)
        self.reward = torch.empty(buffer_size, dtype=torch.float)
        self.next_state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.done = torch.empty(buffer_size, dtype=torch.int)

        self.count = 0
        self.real_size = 0

    def add(self, transition):
        state, action, reward, next_state, done = transition

        self.state[self.count] = torch.as_tensor(state)
        self.action[self.count] = torch.as_tensor(action)
        self.reward[self.count] = torch.as_tensor(reward)
        self.next_state[self.count] = torch.as_tensor(next_state)
        self.done[self.count] = torch.as_tensor(done)

        self.count = (self.count + 1) % self.size
        self.real_size = min(self.real_size + 1, self.size)

    def sample(self, batch_size, current_state):
        assert self.real_size >= batch_size, "Buffer has fewer samples than batch size"

        # Sample lambda * k transitions
        k = batch_size
        lambda_k = self.lambda_factor * k
        indices = np.random.choice(self.real_size, lambda_k, replace=False)

        # Compute similarity between sampled states and current state
        current_state_tensor = torch.tensor(current_state, dtype=torch.float).view(1, -1)
        sampled_states = self.state[indices]
        similarities = torch.nn.functional.cosine_similarity(current_state_tensor, sampled_states)

        # Select top-k similar indices
        topk_indices = similarities.topk(k).indices
        final_indices = torch.tensor(indices)[topk_indices]

        batch = (
            self.state[final_indices].to(device),
            self.action[final_indices].to(device),
            self.reward[final_indices].to(device),
            self.next_state[final_indices].to(device),
            self.done[final_indices].to(device)
        )
        return batch