import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math 
import numpy as np 
from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('state', 'pitch_action', 'yaw_action', 'next_state', 'pitch_reward', 'yaw_reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def get(self, index):
        return [self.memory[index]]

    def __len__(self):
        return len(self.memory)


class DQNNetwork(nn.Module):

    def __init__(self, history, n_pitch_actions, n_yaw_actions):
        super(DQNNetwork, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(in_features= history * 2, out_features= 512),
            nn.ReLU(),
            nn.Linear(in_features= 512, out_features= 256),
            nn.ReLU(),           
        )

        self.yaw_fc = nn.Sequential(
            nn.Linear(in_features= 256, out_features= 128),
            nn.ReLU(),
            nn.Linear(in_features= 128, out_features= n_yaw_actions)
        )

        self.pitch_fc = nn.Sequential(
            nn.Linear(in_features= 256, out_features= 128),
            nn.ReLU(),
            nn.Linear(in_features= 128, out_features= n_pitch_actions)
        )


    # Called with either one element to determine next action, or a batch
    # during optimization.
    def forward(self, s):
        shared = self.fc(s)
        pitch = self.pitch_fc(shared)
        yaw = self.yaw_fc(shared)
        return pitch, yaw
    
class DQN:

    def __init__(self, n_pitch_actions, n_yaw_actions, history_size) -> None:
        self.BATCH_SIZE = 32
        self.GAMMA = 0.99
        self.EPS_START = 0.8
        self.EPS_END = 0.1
        self.EPS_DECAY = 50000
        self.TAU = 0.002
        LR = 1e-4
        self.MEMORY_SIZE = 50000
        self.n_pitch_actions = n_pitch_actions
        self.n_yaw_actions = n_yaw_actions
        self.history_size = history_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQNNetwork(self.history_size, self.n_pitch_actions, self.n_yaw_actions).to(self.device)
        self.target_net = DQNNetwork(self.history_size, self.n_pitch_actions, self.n_yaw_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(self.MEMORY_SIZE)
        self.steps_done = 0
        
    def select_action(self, s):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                pitch, yaw = self.policy_net(s)
                return pitch.max(1)[1].view(1, 1), yaw.max(1)[1].view(1, 1)
        else:
            return torch.tensor([[int(np.random.randint(0,self.n_pitch_actions))]], device=self.device, dtype=torch.long), torch.tensor([[int(np.random.randint(0,self.n_yaw_actions))]], device=self.device, dtype=torch.long)
    
    def select_eval_action(self, s):
        with torch.no_grad():
            pitch, yaw = self.target_net(s)
            return pitch.max(1)[1].view(1, 1), yaw.max(1)[1].view(1, 1)

    def optimize(self):
        if len(self.memory) < self.BATCH_SIZE:
            return None
        self.steps_done += 1
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                        batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        
        state_batch = torch.cat(batch.state) #S
        yaw_action_batch = torch.cat(batch.yaw_action) #A
        pitch_action_batch = torch.cat(batch.pitch_action) #A
        yaw_reward_batch = torch.cat(batch.yaw_reward) #R
        pitch_reward_batch = torch.cat(batch.pitch_reward) #R

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        pitch, yaw  = self.policy_net(state_batch)
        state_action_values_pitch = pitch.gather(1, pitch_action_batch)
        state_action_values_yaw = yaw.gather(1, yaw_action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values_pitch = torch.zeros(self.BATCH_SIZE, device=self.device)
        next_state_values_yaw = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            #DQN
            ns_pitch, ns_yaw = self.target_net(non_final_next_states)
            next_state_values_pitch[non_final_mask] = ns_pitch.max(1)[0]
            next_state_values_yaw[non_final_mask] = ns_yaw.max(1)[0]

            #DDQN
            # qt = self.target_net(non_final_next_states)
            # qp = self.policy_net(non_final_next_states)
            # next_state_values_pitch[non_final_mask] = qt[0].gather(1, qp[0].argmax(1).reshape(1,-1)) #Qt(S_t+1, argmax_a Qp(S_t+1,a))
            # next_state_values_yaw[non_final_mask] = qt[1].gather(1, qp[1].argmax(1).reshape(1,-1)) #Qt(S_t+1, argmax_a Qp(S_t+1,a))

        # Compute the expected Q values
        expected_state_action_values_pitch = (next_state_values_pitch * self.GAMMA) + pitch_reward_batch  #R + gamma * target
        expected_state_action_values_yaw = (next_state_values_yaw * self.GAMMA) + yaw_reward_batch  #R + gamma * target

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        pitch_loss = criterion(state_action_values_pitch, expected_state_action_values_pitch.unsqueeze(1)) 

        criterion = nn.SmoothL1Loss()
        yaw_loss = criterion(state_action_values_yaw, expected_state_action_values_yaw.unsqueeze(1)) 

        loss = pitch_loss + yaw_loss
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        return loss