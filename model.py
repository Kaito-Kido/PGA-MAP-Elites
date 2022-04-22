import copy
from enum import auto
import numpy as np
from networks import CriticNetwork
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from utils import soft_update, hard_update

import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, 256)
        self.linear5 = nn.Linear(256, 256)
        self.linear6 = nn.Linear(256, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, 256)
        self.linear2 = nn.Linear(256, 256)

        self.mean_linear = nn.Linear(256, num_actions)
        self.log_std_linear = nn.Linear(256, num_actions)

        self.apply(weights_init_)

        self.type = None
        self.id = None
        self.parent_1_id = None
        self.parent_2_id = None
        self.novel = None
        self.delta_f = None

        for param in self.parameters():
            param.requires_grad = False

        # action rescaling
        # if action_space is None:
        #     self.action_scale = torch.tensor(1.)
        #     self.action_bias = torch.tensor(0.)
        # else:
        #     self.action_scale = torch.FloatTensor(
        #         (action_space.high - action_space.low) / 2.)
        #     self.action_bias = torch.FloatTensor(
        #         (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action, _, _ = self.sample(state)
        return action.cpu().data.numpy()[0]

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(
            filename, map_location=torch.device('cpu')))

    def disable_grad(self):
        for param in self.paramteres():
            param.requires_grad = False

    def enable_grad(self):
        for param in self.parameters():
            param.requires_grad = True

    def return_copy(self):
        return copy.deepcopy(self)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, 256)
        self.linear2 = nn.Linear(256, 256)

        self.mean = nn.Linear(256, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)


class Critic(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        policy_type="Gaussian",
        target_update_interval=1,
        automatic_entropy_tuning=False,
    ):

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.policy_type = policy_type
        self.target_update_interval = target_update_interval
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.total_it = 0
        self.actors_set = set()
        self.actors = []
        self.actor_targets = []
        self.actor_optimisers = []

        self.critic = CriticNetwork(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.paramers(), lr=3e-4)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = - \
                    torch.prod(torch.Tensor(
                        action_dim.shape).to(device)).item()
                self.log_alpha = torch.zeros(
                    1, requires_grad=True, device=device)
                self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=3e-4)

            self.policy = GaussianPolicy(
                state_dim, action_dim.shape[0], 26, action_dim).to(device)
            self.policy_optim = torch.optim.Adam(
                self.policy.parameters(), lr=3e-4)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(
                state_dim, action_dim.shape[0], 256, action_dim).to(self.device)
            self.policy_optim = torch.optim.Adam(
                self.policy.parameters(), lr=3e-4)

    def train(self, archive, replay_buffer, nr_of_steps, batch_size=256):
        diff = set(archive.keys()) - self.actors_set
        for desc in diff:
            self.actors_set.add(desc)
            new_actor = archive[desc].x
            a = copy.deepcopy(new_actor)
            for param in a.parameters():
                param.requires_grad = True
            a.parent_1_id = new_actor.id
            a.parent_2_id = None
            a.type = "critc_training"
            target = copy.deepcopy(a)
            optimizer = torch.optim.Adam(a.parameters(), lr=3e-4)
            self.actors.append(a)
            self.actor_targets.append(target)
            self.actor_optimisers.append(optimizer)

        for _ in range(nr_of_steps):
            self.total_it += 1

            state, action, next_state, reward, not_done = replay_buffer.sample(
                batch_size)

            all_target_Q = torch.zeros(batch_size, len(self.actors))
            with torch.no_grad():
                for idx, actor in enumerate(self.actors):
                    next_action, next_state_log_pi, _ = self.actor_targets[idx].sample(
                        next_state)
                    target_Q1, target_Q2 = self.critic_target(
                        next_state, next_action)
                    target_Q = torch.min(
                        target_Q1, target_Q2) - self.alpha * next_state_log_pi
                    target_Q[:, idx] = target_Q.squeeze()

                target_Q = torch.max(all_target_Q, dim=1, keepdim=True)[0]
                target_Q = reward + not_done * self.gamma * target_Q

            current_Q1, current_Q2 = self.critic(state, action)

            critic_loss = F.mse_loss(
                current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            for idx, actor in enumerate(self.actors):

                pi, log_pi, _ = actor.sample(state)

                qf1_pi, qf2_pi = self.critic(state, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)

                actor_loss = (self.alpha * log_pi - min_qf_pi).mean()

                self.actor_optimizers[idx].zero_grad()
                actor_loss.backward()
                self.actor_optimisers[idx].step()

                for param, target_param in zip(actor.parameters(), self.actor_targets[idx].parameters()):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data)

            # for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            #     target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            if nr_of_steps % self.target_update_interval == 0:
                soft_update(self.critic_target, self.critic, self.tau)

        return critic_loss

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename)
        torch.save(self.critic_optimizer.state_dict(), filename + "_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(
            filename + "_critic", map_location=torch.device('cpu')))
        self.critic_optimizer.load_state_dict(torch.load(
            filename + "_critic_optimizer", map_location=torch.device('cpu')))
        self.critic_target = copy.deepcopy(self.critic)
