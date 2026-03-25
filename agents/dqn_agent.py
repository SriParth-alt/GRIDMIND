"""
dqn_agent.py
Improvements over v1:
  - Dueling DQN architecture  (value + advantage streams)
  - Prioritised Experience Replay (PER) — trains more on surprising transitions
  - Target network with soft update (Polyak averaging) instead of hard copy
  - Gradient clipping and weight decay for stable training
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


# ──────────────────────────────────────────────────────────────────────── #
#  Dueling DQN                                                             #
# ──────────────────────────────────────────────────────────────────────── #
class DuelingDQN(nn.Module):
    """
    Dueling architecture separates state value V(s) from action advantages A(s,a).
    Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
    This stabilises learning when many actions have similar Q-values.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.value_stream     = nn.Linear(128, 1)
        self.advantage_stream = nn.Linear(128, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.shared(x)
        V    = self.value_stream(feat)                          # (B, 1)
        A    = self.advantage_stream(feat)                      # (B, actions)
        Q    = V + A - A.mean(dim=1, keepdim=True)             # dueling combination
        return Q


# ──────────────────────────────────────────────────────────────────────── #
#  Prioritised Replay Buffer                                               #
# ──────────────────────────────────────────────────────────────────────── #
class PrioritisedReplayBuffer:
    """
    Stores (s,a,r,s',done) with a priority = |TD-error| + epsilon.
    High-error transitions are sampled more often → faster learning.
    """
    def __init__(self, maxlen: int = 20000, alpha: float = 0.6):
        self.buffer    = deque(maxlen=maxlen)
        self.priorities = deque(maxlen=maxlen)
        self.alpha      = alpha          # how much to use priority (0=uniform)

    def push(self, *transition):
        max_p = max(self.priorities) if self.priorities else 1.0
        self.buffer.append(transition)
        self.priorities.append(max_p)

    def sample(self, batch_size: int, beta: float = 0.4):
        probs = np.array(self.priorities, dtype=np.float64) ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        samples = [self.buffer[i] for i in indices]

        # Importance-sampling weights to correct for biased sampling
        n = len(self.buffer)
        weights = (n * probs[indices]) ** (-beta)
        weights /= weights.max()

        return samples, indices, torch.FloatTensor(weights)

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = abs(err) + 1e-5

    def __len__(self):
        return len(self.buffer)


# ──────────────────────────────────────────────────────────────────────── #
#  Agent                                                                   #
# ──────────────────────────────────────────────────────────────────────── #
class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int):
        self.action_dim = action_dim
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model        = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_model = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=0.0005, weight_decay=1e-5
        )
        self.criterion = nn.SmoothL1Loss(reduction="none")   # Huber loss per sample

        self.gamma          = 0.97      # longer horizon — relevant for multi-hour dispatch
        self.tau            = 0.005     # Polyak soft-update coefficient

        # Epsilon: decays to 0.01 in exactly 1500 episodes
        # decay = (0.01 / 1.0) ^ (1/1500) ≈ 0.9969
        self.epsilon        = 1.0
        self.epsilon_decay  = 0.9969
        self.epsilon_min    = 0.01

        self.memory         = PrioritisedReplayBuffer(maxlen=20000)
        self.batch_size     = 64
        self.min_replay     = 256       # don't train until buffer has this many samples
        self.beta           = 0.4       # IS weight annealing start
        self.beta_increment = 0.001     # anneal beta toward 1.0 over training

        self.episode_count  = 0

    # ──────────────────────── #
    def choose_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model(t).argmax(dim=1).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    # ──────────────────────── #
    def train(self):
        if len(self.memory) < self.min_replay:
            return None

        self.beta = min(1.0, self.beta + self.beta_increment)
        samples, indices, weights = self.memory.sample(self.batch_size, self.beta)

        states, actions, rewards, next_states, dones = zip(*samples)

        states      = torch.FloatTensor(np.array(states)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        actions     = torch.LongTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        dones       = torch.FloatTensor(dones).to(self.device)
        weights     = weights.to(self.device)

        # Current Q
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target: online net picks action, target net evaluates it
        with torch.no_grad():
            best_actions  = self.model(next_states).argmax(1)
            next_q        = self.target_model(next_states).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            targets       = rewards + self.gamma * next_q * (1 - dones)

        td_errors = (targets - q_values).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)

        # Weighted Huber loss
        loss = (weights * self.criterion(q_values, targets)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Soft update target network (Polyak)
        for tp, op in zip(self.target_model.parameters(), self.model.parameters()):
            tp.data.copy_(self.tau * op.data + (1 - self.tau) * tp.data)

        return loss.item()

    # ──────────────────────── #
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episode_count += 1