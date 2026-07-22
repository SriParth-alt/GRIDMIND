"""
dqn_agent.py
Dueling DQN with:
  - Prioritised Experience Replay (PER)
  - Double DQN target computation
  - Polyak soft target network update
  - Gradient clipping + weight decay
All hyperparameters pulled from config.py.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

from config import (
    LEARNING_RATE, WEIGHT_DECAY, GAMMA, TAU,
    BATCH_SIZE, MIN_REPLAY, REPLAY_MAXLEN,
    PER_ALPHA, BETA_START, BETA_INCREMENT,
    EPSILON_START, EPSILON_MIN, EPSILON_DECAY,
)


# ──────────────────────────────────────────────────────────────────────── #
#  Dueling DQN                                                             #
# ──────────────────────────────────────────────────────────────────────── #
class DuelingDQN(nn.Module):
    """Q(s,a) = V(s) + A(s,a) - mean(A(s,.))"""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),       nn.ReLU(),
        )
        self.value_stream     = nn.Linear(128, 1)
        self.advantage_stream = nn.Linear(128, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.shared(x)
        V    = self.value_stream(feat)
        A    = self.advantage_stream(feat)
        return V + A - A.mean(dim=1, keepdim=True)


# ──────────────────────────────────────────────────────────────────────── #
#  Prioritised Replay Buffer                                               #
# ──────────────────────────────────────────────────────────────────────── #
class PrioritisedReplayBuffer:
    """Transitions with higher TD-error are sampled more often."""

    def __init__(self, maxlen: int = REPLAY_MAXLEN, alpha: float = PER_ALPHA):
        self.buffer     = deque(maxlen=maxlen)
        self.priorities = deque(maxlen=maxlen)
        self.alpha      = alpha

    def push(self, *transition):
        max_p = max(self.priorities) if self.priorities else 1.0
        self.buffer.append(transition)
        self.priorities.append(max_p)

    def sample(self, batch_size: int, beta: float = BETA_START):
        probs = np.array(self.priorities, dtype=np.float64) ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        samples = [self.buffer[i] for i in indices]

        n       = len(self.buffer)
        weights = (n * probs[indices]) ** (-beta)
        weights /= weights.max()

        return samples, indices, torch.FloatTensor(weights.astype(np.float32))

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            self.priorities[int(idx)] = float(abs(err)) + 1e-5

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
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        self.criterion = nn.SmoothL1Loss(reduction="none")

        self.gamma          = GAMMA
        self.tau            = TAU
        self.epsilon        = EPSILON_START
        self.epsilon_decay  = EPSILON_DECAY
        self.epsilon_min    = EPSILON_MIN

        self.memory         = PrioritisedReplayBuffer()
        self.batch_size     = BATCH_SIZE
        self.min_replay     = MIN_REPLAY
        self.beta           = BETA_START
        self.beta_increment = BETA_INCREMENT

        self.episode_count  = 0

    def choose_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model(t).argmax(dim=1).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def train(self):
        if len(self.memory) < self.min_replay:
            return None

        self.beta = min(1.0, self.beta + self.beta_increment)
        samples, indices, weights = self.memory.sample(self.batch_size, self.beta)

        states, actions, rewards, next_states, dones = zip(*samples)

        states      = torch.FloatTensor(np.array(states,      dtype=np.float32)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states, dtype=np.float32)).to(self.device)
        actions     = torch.LongTensor(list(actions)).to(self.device)
        rewards     = torch.FloatTensor(list(rewards)).to(self.device)
        dones       = torch.FloatTensor(list(dones)).to(self.device)
        weights     = weights.to(self.device)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: online net selects action, target net evaluates it
        with torch.no_grad():
            best_actions = self.model(next_states).argmax(1)
            next_q       = self.target_model(next_states).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            targets      = rewards + self.gamma * next_q * (1 - dones)

        td_errors = (targets - q_values).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)

        loss = (weights * self.criterion(q_values, targets)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Polyak soft-update target network
        for tp, op in zip(self.target_model.parameters(), self.model.parameters()):
            tp.data.copy_(self.tau * op.data + (1 - self.tau) * tp.data)

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episode_count += 1
