"""
q_learning.py
=============
Tabular Q-Learning agent for the 3×3 penalty environment.

Q-table shape: (n_states=16, n_actions=15)
  - state  = previous shot position (0-14, 15=initial)
  - action = cell to dive to (0-14)
"""

import numpy as np
import random
import pickle
import os


class QLearningAgent:
    def __init__(
        self,
        n_states:    int   = 16,
        n_actions:   int   = 15,
        alpha:       float = 0.1,    # learning rate
        gamma:       float = 0.9,    # discount factor
        epsilon:     float = 1.0,    # initial exploration rate
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        self.n_states      = n_states
        self.n_actions     = n_actions
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Initialise Q-table with zeros
        self.Q = np.zeros((n_states, n_actions))

        # Tracking
        self.episode_rewards = []
        self.episode_outcomes = []

    # ------------------------------------------------------------------ #
    #  Action selection                                                    #
    # ------------------------------------------------------------------ #

    def select_action(self, state: int, greedy: bool = False) -> int:
        """
        Epsilon-greedy action selection.
        Set `greedy=True` for evaluation / play mode (no exploration).
        """
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        # Break ties randomly
        max_val = np.max(self.Q[state])
        best_actions = np.where(self.Q[state] == max_val)[0]
        return int(np.random.choice(best_actions))

    # ------------------------------------------------------------------ #
    #  Learning update                                                     #
    # ------------------------------------------------------------------ #

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool):
        """Standard Q-learning (Bellman) update."""
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.Q[next_state])

        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error

    def decay_epsilon(self):
        """Call once per episode to reduce exploration over time."""
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, path: str = "models/q_table.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "Q":               self.Q,
                "epsilon":         self.epsilon,
                "episode_rewards": self.episode_rewards,
                "episode_outcomes": self.episode_outcomes,
            }, f)

    def load(self, path: str = "models/q_table.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.Q                = data["Q"]
        self.epsilon          = data["epsilon"]
        self.episode_rewards  = data.get("episode_rewards", [])
        self.episode_outcomes = data.get("episode_outcomes", [])

    # ------------------------------------------------------------------ #
    #  Diagnostics                                                         #
    # ------------------------------------------------------------------ #

    def best_actions(self) -> dict:
        """Return the greedy best action for every state."""
        return {s: int(np.argmax(self.Q[s])) for s in range(self.n_states)}

    def __repr__(self):
        return (f"QLearningAgent(α={self.alpha}, γ={self.gamma}, "
                f"ε={self.epsilon:.3f})")