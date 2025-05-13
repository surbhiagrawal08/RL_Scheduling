import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Iterable
from itertools import chain
from typing import SupportsFloat
from torch import Tensor
import numpy as np
from torch.utils.data import Dataset, DataLoader

from .trainer import Trainer

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, hidden_dim, num_actions):
        super().__init__()
        self.fc = nn.Linear(obs_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, num_actions)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        return F.softmax(self.actor(x), dim=-1), self.critic(x)


import torch.multiprocessing as mp

def actor_process(actor_id, global_model, queue, env_fn):
    env = env_fn()
    local_model = ActorCritic(
        obs_dim=env.observation_space.shape[0],
        hidden_dim=128,
        num_actions=env.action_space.n,
    )
    while True:
        local_model.load_state_dict(global_model.state_dict())  # sync
        traj = collect_trajectory(env, local_model)
        queue.put(traj)

def compute_returns(rewards, gamma=0.99):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32)

def learner(queue, global_model, optimizer, batch_size=8):
    while True:
        batch_states, batch_actions, batch_returns, batch_values = [], [], [], []

        for _ in range(batch_size):
            traj = queue.get()
            states, actions, rewards, values = zip(*traj)
            returns = compute_returns(rewards)
            batch_states += states
            batch_actions += actions
            batch_returns += list(returns)
            batch_values += list(values)

        states = torch.tensor(batch_states, dtype=torch.float32)
        actions = torch.tensor(batch_actions, dtype=torch.int64)
        returns = torch.tensor(batch_returns, dtype=torch.float32)
        values = torch.tensor(batch_values, dtype=torch.float32)

        probs, critic_vals = global_model(states)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        advantage = returns - critic_vals.squeeze()

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
