import torch
from torch import nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(state_dim, hidden_size)
        self.layer2 = nn.Linear(hidden_size, action_dim)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        return self.layer2(x)
    
    
    