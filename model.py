import torch as T
import torch.nn as nn
from torch.distributions import Normal, Categorical


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0, categoric=False):
        super(ActorCritic, self).__init__()

        self.categoric = categoric
        self.num_outputs = num_outputs

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs)
        )
        
        #random factor
        if not categoric:
            self.log_std = nn.Parameter(T.ones(1, num_outputs) * std)

    def forward(self, x):
        value = self.critic(x)
        mu = self.actor(x)

        if not self.categoric:
            std = self.log_std.exp().expand_as(mu)
            dist = Normal(mu, std)
        else:
            dist = Categorical(mu)
        
        return dist, value