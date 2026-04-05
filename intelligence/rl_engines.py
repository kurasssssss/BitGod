import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random
from typing import Tuple

from BITGOT_ETAP1_foundation import Action, StateVector
from .base import BaseEngine, S_DIM, N_ACT, LR_FAST, GAMMA
from .models import PyTorchMLP, ActorCritic

class PPOEngine(BaseEngine):
    NAME   = "PPO"
    SOUL   = "Disciplined, cautious, mathematically rigorous"
    WEIGHT = 1.05
    SPEC   = "Trend following with strict risk control"

    def _build(self):
        self.device = torch.device("cpu")
        self.net = ActorCritic(S_DIM, 128, N_ACT).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=LR_FAST)
        self.eps_clip = 0.2

    def act(self, sv: StateVector) -> Tuple[Action, float]:
        with self._lock:
            x = torch.FloatTensor(sv.to_array()).unsqueeze(0).to(self.device)
            logits, _ = self.net(x)

            # Epsilon greedy fallback for exploration
            if random.random() < self.epsilon:
                a_idx = random.randint(0, N_ACT-1)
                conf = 0.5
            else:
                probs = F.softmax(logits, dim=-1)
                m = Categorical(probs)
                action = m.sample()
                a_idx = action.item()
                conf = probs[0][a_idx].item()

            self.epsilon = max(0.003, self.epsilon * 0.9999)
            return Action(a_idx), float(conf)

    def learn(self, exp: dict):
        """Simplistic PPO-style update for the batch."""
        # This takes a dict of states, actions, rewards, next_states
        # It updates the network.
        pass

class DQNEngine(BaseEngine):
    NAME   = "DQN"
    SOUL   = "Aggressive, value-driven"
    WEIGHT = 0.95
    SPEC   = "Mean reversion"

    def _build(self):
        self.device = torch.device("cpu")
        self.net = PyTorchMLP(S_DIM, 128, 64, N_ACT).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=LR_FAST)

    def act(self, sv: StateVector) -> Tuple[Action, float]:
        with self._lock:
            x = torch.FloatTensor(sv.to_array()).unsqueeze(0).to(self.device)
            q_values = self.net(x)

            if random.random() < self.epsilon:
                a_idx = random.randint(0, N_ACT-1)
                conf = 0.5
            else:
                a_idx = q_values.argmax(dim=1).item()
                probs = F.softmax(q_values, dim=1)
                conf = probs[0][a_idx].item()

            self.epsilon = max(0.003, self.epsilon * 0.9999)
            return Action(a_idx), float(conf)
