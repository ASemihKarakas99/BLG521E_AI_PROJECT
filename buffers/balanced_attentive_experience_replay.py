import torch
import numpy as np

class BalancedAttentiveReplayBuffer:
    def __init__(self, state_size, action_size, buffer_size, lambda_factor=2, diversity_ratio=0.2):
        self.lambda_factor = lambda_factor
        self.diversity_ratio = diversity_ratio
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

    def sample(self, batch_size, current_state, embedding_model=None):
        assert self.real_size >= batch_size, "Buffer has fewer samples than batch size"

        k = batch_size
        attentive_k = int((1 - self.diversity_ratio) * k)
        random_k = k - attentive_k
        lambda_k = min(self.lambda_factor * k, self.real_size)

        # Sample candidate set for attention
        candidate_indices = np.random.choice(self.real_size, lambda_k, replace=False)
        candidate_states = self.state[candidate_indices]

        # Compute similarity
        with torch.no_grad():
            current_state_tensor = torch.tensor(current_state, dtype=torch.float).view(1, -1)
            if embedding_model:
                current_embed = embedding_model.get_embedding(current_state_tensor)
                candidate_embeds = embedding_model.get_embedding(candidate_states)
            else:
                current_embed = current_state_tensor
                candidate_embeds = candidate_states

            similarities = torch.nn.functional.cosine_similarity(current_embed, candidate_embeds)

        topk_indices = similarities.topk(attentive_k).indices
        attentive_indices = torch.tensor(candidate_indices[topk_indices])

        # Sample diverse indices uniformly
        diverse_indices = np.random.choice(self.real_size, random_k, replace=False)

        # Merge both
        final_indices = torch.cat([attentive_indices, torch.tensor(diverse_indices)])

        # Compute importance sampling weights (uniform for now, bias correction via annealing possible later)
        probs = torch.ones(k) / self.real_size  # Uniform sampling probability
        is_weights = (1.0 / (self.real_size * probs)).pow(0.6)  # β=0.6
        is_weights /= is_weights.max()  # Normalize

        batch = (
            self.state[final_indices],
            self.action[final_indices],
            self.reward[final_indices],
            self.next_state[final_indices],
            self.done[final_indices]
        )
        return batch, is_weights.to(self.state.device)

