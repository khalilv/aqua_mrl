import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math 
import numpy as np 
from collections import namedtuple, deque
import torch.nn.init as init

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
            nn.Linear(in_features= 256, out_features= 128),
            nn.ReLU(),  
        )

        self.yaw_fc = nn.Sequential(
            nn.Linear(in_features= 128, out_features= n_yaw_actions)
        )

        self.pitch_fc = nn.Sequential(
            nn.Linear(in_features= 128, out_features= n_pitch_actions)
        )

        # Apply He initialization to linear layers
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                init.zeros_(m.bias)


    # Called with either one element to determine next action, or a batch
    # during optimization.
    def forward(self, s):
        shared = self.fc(s)
        pitch = self.pitch_fc(shared)
        yaw = self.yaw_fc(shared)
        return pitch, yaw
    
class DQN:

    def __init__(self, n_pitch_actions, n_yaw_actions, history_size) -> None:
        self.BATCH_SIZE = 64
        self.GAMMA = 0.9
        self.EPS_START = 0.9
        self.EPS_END = 0.1
        self.EPS_DECAY = 250000
        self.TAU = 0.0025
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
        td_estimate_pitch = pitch.gather(1, pitch_action_batch)
        td_estimate_yaw = yaw.gather(1, yaw_action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values_pitch = torch.zeros(self.BATCH_SIZE, device=self.device)
        next_state_values_yaw = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            # #DQN
            # ns_pitch, ns_yaw = self.target_net(non_final_next_states)
            # next_state_values_pitch[non_final_mask] = ns_pitch.max(1)[0]
            # next_state_values_yaw[non_final_mask] = ns_yaw.max(1)[0]

            #DDQN
            ns_target_pitch, ns_target_yaw = self.target_net(non_final_next_states)
            ns_policy_pitch, ns_policy_yaw = self.policy_net(non_final_next_states)
            best_action_pitch = torch.argmax(ns_policy_pitch, axis=1).unsqueeze(0).t()
            best_action_yaw = torch.argmax(ns_policy_yaw, axis=1).unsqueeze(0).t()

            next_state_values_pitch[non_final_mask] = ns_target_pitch.gather(1, best_action_pitch).squeeze() #Qt(S_t+1, argmax_a Qp(S_t+1,a))
            next_state_values_yaw[non_final_mask] = ns_target_yaw.gather(1, best_action_yaw).squeeze() #Qt(S_t+1, argmax_a Qp(S_t+1,a))
           
        # Compute the expected Q values
        td_target_pitch = (next_state_values_pitch * self.GAMMA) + pitch_reward_batch  #R + gamma * target
        td_target_yaw = (next_state_values_yaw * self.GAMMA) + yaw_reward_batch  #R + gamma * target

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        pitch_loss = criterion(td_estimate_pitch, td_target_pitch.unsqueeze(1)) 
        yaw_loss = criterion(td_estimate_yaw, td_target_yaw.unsqueeze(1)) 

        loss = pitch_loss + yaw_loss
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        return loss