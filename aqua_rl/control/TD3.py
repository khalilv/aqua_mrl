import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import random
from collections import namedtuple, deque


# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)
	
	def add_batch(self, states, actions, next_states, rewards, not_dones, n):
		self.state[self.ptr:self.ptr+n] = states
		self.action[self.ptr:self.ptr+n] = actions
		self.next_state[self.ptr:self.ptr+n] = next_states
		self.reward[self.ptr:self.ptr+n] = rewards
		self.not_done[self.ptr:self.ptr+n] = not_dones
		self.ptr = (self.ptr + n) % self.max_size
		self.size = min(self.size + n, self.max_size)

class Actor(nn.Module):
	
	def __init__(self, history, action_dim, limit):
		super(Actor, self).__init__()

		self.fc = nn.Sequential(
            nn.Linear(in_features= history * 3, out_features= 256),
            nn.ReLU(),
            nn.Linear(in_features= 256, out_features= 256),
            nn.ReLU(),  
            nn.Linear(in_features= 256, out_features= action_dim)
        )

		self.limit = limit
		
        # Apply He initialization to linear layers
		self.apply(self.init_weights)
		
	def init_weights(self, m):
		if isinstance(m, nn.Linear):
			init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
			if m.bias is not None:
				init.zeros_(m.bias)

	def forward(self, s):
		s = self.fc(s)
		return self.limit * torch.tanh(s)


class Critic(nn.Module):
	def __init__(self, history, action_dim):
		super(Critic, self).__init__()

		self.q1 = nn.Sequential(
            nn.Linear(in_features= history * 3 + action_dim, out_features= 256),
            nn.ReLU(),
            nn.Linear(in_features= 256, out_features= 256),
            nn.ReLU(),  
            nn.Linear(in_features= 256, out_features= 1)
        )

		self.q2 = nn.Sequential(
            nn.Linear(in_features= history * 3 + action_dim, out_features= 256),
            nn.ReLU(),
            nn.Linear(in_features= 256, out_features= 256),
            nn.ReLU(),  
            nn.Linear(in_features= 256, out_features= 1)
        )
		
        # Apply He initialization to linear layers
		self.apply(self.init_weights)
		
	def init_weights(self, m):
		if isinstance(m, nn.Linear):
			init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
			if m.bias is not None:
				init.zeros_(m.bias)

	def forward(self, s, a):
		sa = torch.cat((s, a), 1)
		return self.q1(sa), self.q2(sa)

	def Q1(self,s,a):
		sa = torch.cat((s, a), 1)
		return self.q1(sa)


class TD3(object):
	def __init__(
		self,
		history,
		action_dim,
		limit,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		expl_noise = 0.1,
		capacity = 50000
	):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.actor = Actor(history, action_dim, limit).to(self.device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(history, action_dim).to(self.device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.limit = limit
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq
		self.expl_noise = expl_noise
		self.memory_capacity = capacity
		self.memory = ReplayBuffer(int(history*3), action_dim, self.memory_capacity)
		self.total_it = 0


	def select_action(self, s):
		return self.actor(s)
	
	def train(self, batch_size=64):
		if self.memory.size < batch_size:
			return None
		
		self.total_it += 1

        # Sample replay buffer 
		state, action, next_state, reward, not_done = self.memory.sample(batch_size)
		
		with torch.no_grad():
            # Select action according to policy and add clipped noise
			noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
			next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.limit, self.limit)

            # Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

        # Delayed policy updates
		if self.total_it % self.policy_freq == 0:

            # Compute actor losse
			actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
            # Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

            # Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
				
			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
				
		return critic_loss
