import torch
import torch.nn as nn
import torch.nn.functional as F

class PyTorchMLP(nn.Module):
    """
    Standard PyTorch MLP replacing the old NumpyMLP.
    Includes proper parameter initialization and dropout.
    """
    def __init__(self, input_dim: int, hidden_dim1: int, hidden_dim2: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

class ActorCritic(nn.Module):
    """Base network for PPO and A3C"""
    def __init__(self, input_dim, hidden_dim, action_dim):
        super().__init__()
        # Shared features
        self.base = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim)
        )
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        features = self.base(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value
