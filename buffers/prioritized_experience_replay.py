import numpy as np
import random
from collections import deque
from typing import List, Tuple, Any

class SumTree:
    def __init__(self, capacity: int):
        """
        Initialize SumTree data structure.
        
        Args:
            capacity: Maximum number of leaf nodes (experiences)
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.position = 0
        
    def _propagate(self, idx: int, change: float) -> None:
        """
        Propagate priority change up the tree.
        
        Args:
            idx: Index of the leaf node
            change: Change in priority value
        """
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
            
    def _retrieve(self, idx: int, s: float) -> int:
        """
        Retrieve leaf node index based on priority value.
        
        Args:
            idx: Current node index
            s: Priority value to search for
            
        Returns:
            Index of the leaf node
        """
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
            
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
            
    def total(self) -> float:
        """Return total priority."""
        return self.tree[0]
        
    def add(self, priority: float, data: Any) -> None:
        """
        Add new experience to the tree.
        
        Args:
            priority: Priority value
            data: Experience data
        """
        idx = self.position + self.capacity - 1
        self.data[self.position] = data
        self.update(idx, priority)
        
        self.position = (self.position + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1
            
    def update(self, idx: int, priority: float) -> None:
        """
        Update priority of a leaf node.
        
        Args:
            idx: Index of the leaf node
            priority: New priority value
        """
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
        
    def get(self, s: float) -> Tuple[int, Any, float]:
        """
        Get experience based on priority value.
        
        Args:
            s: Priority value to search for
            
        Returns:
            Tuple of (tree index, experience data, priority)
        """
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.data[data_idx], self.tree[idx]

class PrioritizedExperienceReplay:
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6
    ):
        """
        Initialize Prioritized Experience Replay buffer.
        
        Args:
            capacity: Maximum size of the buffer
            alpha: Priority exponent (0 = uniform sampling, 1 = fully prioritized)
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
            beta_increment: Rate at which beta increases to 1
            epsilon: Small constant to prevent zero priorities
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        
        # Initialize SumTree
        self.tree = SumTree(capacity)
        self.max_priority = 1.0
        
    def add(self, experience: Tuple[Any, ...], error: float = None) -> None:
        """
        Add a new experience to the buffer.
        
        Args:
            experience: Tuple containing (state, action, reward, next_state, done)
            error: TD error for the experience (if None, use max priority)
        """
        if error is None:
            priority = self.max_priority
        else:
            priority = (abs(error) + self.epsilon) ** self.alpha
            
        self.tree.add(priority, experience)
        
    def sample(self, batch_size: int) -> Tuple[List[Tuple], np.ndarray, np.ndarray]:
        """
        Sample a batch of experiences based on their priorities.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple containing:
            - List of experiences
            - Indices of sampled experiences
            - Importance sampling weights
        """
        if self.tree.n_entries < batch_size:
            return None, None, None
            
        experiences = []
        indices = np.zeros(batch_size, dtype=np.int32)
        weights = np.zeros(batch_size, dtype=np.float32)
        
        # Calculate priority segment
        priority_segment = self.tree.total() / batch_size
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        for i in range(batch_size):
            # Sample priority value
            a = priority_segment * i
            b = priority_segment * (i + 1)
            value = np.random.uniform(a, b)
            
            # Get experience
            idx, experience, priority = self.tree.get(value)
            
            # Calculate importance sampling weight
            prob = priority / self.tree.total()
            weight = (self.tree.n_entries * prob) ** (-self.beta)
            weights[i] = weight
            indices[i] = idx
            experiences.append(experience)
            
        # Normalize weights
        weights = weights / np.max(weights)
        
        return experiences, indices, weights
        
    def update_priorities(self, indices: np.ndarray, errors: np.ndarray) -> None:
        """
        Update priorities for experiences based on their TD errors.
        
        Args:
            indices: Indices of experiences to update
            errors: TD errors for the experiences
        """
        priorities = (np.abs(errors) + self.epsilon) ** self.alpha
        for idx, priority in zip(indices, priorities):
            self.tree.update(idx, priority)
        self.max_priority = max(self.max_priority, np.max(priorities))
        
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return self.tree.n_entries
