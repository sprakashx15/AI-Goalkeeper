"""
dqn.py
======
Deep Q-Network (DQN) agent for the penalty environment.

Uses:
  - A small fully-connected network (state → Q-values for all actions)
  - Experience replay (ReplayBuffer)
  - Target network (hard update every N steps)

The state is one-hot encoded before being fed to the network.
"""

import numpy as np
import random
import pickle
import os
from collections import deque

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True

    class QNetwork(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, hidden: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, output_dim),
            )

        def forward(self, x):
            return self.net(x)

except ImportError:
    TORCH_AVAILABLE = False
    QNetwork = None


# ── Replay Buffer ────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int = 10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


# ── DQN Agent ────────────────────────────────────────────────────────────────

class DQNAgent:
    def __init__(
        self,
        n_states:      int   = 16,
        n_actions:     int   = 15,
        lr:            float = 1e-3,
        gamma:         float = 0.9,
        epsilon:       float = 1.0,
        epsilon_min:   float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size:    int   = 64,
        target_update: int   = 50,    # hard-update target net every N episodes
        buffer_size:   int   = 10_000,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for DQNAgent. "
                "Install it with:  pip install torch"
            )

        self.n_states      = n_states
        self.n_actions     = n_actions
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_update = target_update

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.policy_net = QNetwork(n_states, n_actions).to(self.device)
        self.target_net = QNetwork(n_states, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn   = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(buffer_size)

        self.episode_count   = 0
        self.episode_rewards = []
        self.episode_outcomes = []

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _encode(self, state: int) -> np.ndarray:
        """One-hot encode state."""
        v = np.zeros(self.n_states, dtype=np.float32)
        v[state] = 1.0
        return v

    def _to_tensor(self, arr: np.ndarray):
        return torch.FloatTensor(arr).to(self.device)

    # ── Action selection ─────────────────────────────────────────────────────

    def select_action(self, state: int, greedy: bool = False) -> int:
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            s = self._to_tensor(self._encode(state)).unsqueeze(0)
            q = self.policy_net(s)
            return int(q.argmax(dim=1).item())

    # ── Learning update ──────────────────────────────────────────────────────

    def store(self, state, action, reward, next_state, done):
        self.replay_buffer.push(
            self._encode(state), action, reward, self._encode(next_state), done
        )

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        s  = self._to_tensor(states)
        ns = self._to_tensor(next_states)
        a  = torch.LongTensor(actions).to(self.device)
        r  = self._to_tensor(rewards)
        d  = self._to_tensor(dones)

        # Current Q values
        q_vals = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Target Q values
        with torch.no_grad():
            max_next_q = self.target_net(ns).max(dim=1)[0]
            targets    = r + self.gamma * max_next_q * (1 - d)

        loss = self.loss_fn(q_vals, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * self.epsilon_decay)

    def end_episode(self):
        """Call at end of each episode to handle target-net update."""
        self.episode_count += 1
        if self.episode_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str = "models/dqn.pt"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "policy_state":    self.policy_net.state_dict(),
            "target_state":    self.target_net.state_dict(),
            "epsilon":         self.epsilon,
            "episode_count":   self.episode_count,
            "episode_rewards": self.episode_rewards,
            "episode_outcomes": self.episode_outcomes,
        }, path)

    def load(self, path: str = "models/dqn.pt"):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_state"])
        self.target_net.load_state_dict(checkpoint["target_state"])
        self.epsilon          = checkpoint["epsilon"]
        self.episode_count    = checkpoint["episode_count"]
        self.episode_rewards  = checkpoint.get("episode_rewards", [])
        self.episode_outcomes = checkpoint.get("episode_outcomes", [])

    def __repr__(self):
        return (f"DQNAgent(ε={self.epsilon:.3f}, "
                f"buffer={len(self.replay_buffer)}, "
                f"device={self.device})")