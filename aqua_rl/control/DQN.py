import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import namedtuple, deque
import math 
import numpy as np 

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNNetwork(nn.Module):

    def __init__(self, history, n_actions):
        super(DQNNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=history, out_channels=32, kernel_size=5, stride=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features= 3*3*64, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=n_actions),
        )

    # Called with either one element to determine next action, or a batch
    # during optimization.
    def forward(self, x):
        x = self.conv(x)
        x = x.reshape((-1, 3 * 3 * 64))
        x = self.fc(x)
        return x    
    
class DQN:

    def __init__(self, n_actions, history_size) -> None:
        self.BATCH_SIZE = 32
        self.GAMMA = 0.99
        self.EPS_START = 0.9
        self.EPS_END = 0.1
        self.EPS_DECAY = 250000
        self.TAU = 0.002
        LR = 1e-4
        self.MEMORY_SIZE = 100000
        self.n_actions = n_actions
        self.history_size = history_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQNNetwork(self.history_size, self.n_actions).to(self.device)
        self.target_net = DQNNetwork(self.history_size, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.target_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(self.MEMORY_SIZE)
        self.steps_done = 0
        
    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[int(np.random.randint(0,self.n_actions))]], device=self.device, dtype=torch.long)
    
    def select_eval_action(self, state):
        return self.target_net(state).max(1)[1].view(1, 1)

    def optimize(self):
        if len(self.memory) < self.BATCH_SIZE:
            return None
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                        batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        
        state_batch = torch.cat(batch.state) #S
        action_batch = torch.cat(batch.action) #A
        reward_batch = torch.cat(batch.reward) #R

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch) #Qp(S_t,A)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            #DQN
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0] #max_a Qt(S_t+1, a)

            #DDQN
            # self.target_net(non_final_next_states).gather(1, self.policy_net(non_final_next_states).argmax(1).reshape(-1,1)) #Qt(S_t+1, argmax_a Qp(S_t+1,a))
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch  #R + gamma * target

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1)) 
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        return loss