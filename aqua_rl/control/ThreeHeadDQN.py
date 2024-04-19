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
                        ('state', 'x_action', 'y_action', 'z_action', 'next_state', 'reward'))

class ThreeHeadReplayMemory(object):

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


class ThreeHeadDQNNetwork(nn.Module):

    def __init__(self, history, n_x_actions, n_y_actions, n_z_actions):
        super(ThreeHeadDQNNetwork, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(in_features= history * 5 - 2, out_features= 256),
            nn.ReLU(),
            nn.Linear(in_features= 256, out_features= 128),
            nn.ReLU(),  
        )

        self.x_fc = nn.Sequential(
            nn.Linear(in_features= 128, out_features= n_x_actions)
        )

        self.y_fc = nn.Sequential(
            nn.Linear(in_features= 128, out_features= n_y_actions)
        )

        self.z_fc = nn.Sequential(
            nn.Linear(in_features= 128, out_features= n_z_actions)
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
        x = self.x_fc(shared)
        y = self.y_fc(shared)
        z = self.z_fc(shared)
        return x, y, z
    
    
class ThreeHeadDQN:

    def __init__(self, n_x_actions, n_y_actions, n_z_actions, history_size) -> None:
        self.BATCH_SIZE = 128
        self.GAMMA = 0.9
        self.EPS_START = 0.9
        self.EPS_END = 0.1
        self.EPS_DECAY = 100000
        self.TAU = 0.0025
        LR = 1e-4
        self.MEMORY_SIZE = 20000
        self.n_x_actions = n_x_actions
        self.n_y_actions = n_y_actions
        self.n_z_actions = n_z_actions
        self.history_size = history_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = ThreeHeadDQNNetwork(self.history_size, self.n_x_actions, self.n_y_actions, self.n_z_actions).to(self.device)
        self.target_net = ThreeHeadDQNNetwork(self.history_size, self.n_x_actions, self.n_y_actions, self.n_z_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ThreeHeadReplayMemory(self.MEMORY_SIZE)
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
                x, y, z = self.policy_net(s)
                return x.max(1)[1].view(1, 1), y.max(1)[1].view(1, 1), z.max(1)[1].view(1, 1)
        else:
            return torch.tensor([[int(np.random.randint(0,self.n_x_actions))]], device=self.device, dtype=torch.long), torch.tensor([[int(np.random.randint(0,self.n_y_actions))]], device=self.device, dtype=torch.long), torch.tensor([[int(np.random.randint(0,self.n_z_actions))]], device=self.device, dtype=torch.long)
    
    def select_eval_action(self, s):
        with torch.no_grad():
            x, y, z = self.target_net(s)
            return x.max(1)[1].view(1, 1), y.max(1)[1].view(1, 1), z.max(1)[1].view(1, 1)
   
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
        x_action_batch = torch.cat(batch.x_action) #A
        y_action_batch = torch.cat(batch.y_action) #A
        z_action_batch = torch.cat(batch.z_action) #A
        reward_batch = torch.cat(batch.reward) #R

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        x, y, z  = self.policy_net(state_batch)
        td_estimate_x = x.gather(1, x_action_batch)
        td_estimate_y = y.gather(1, y_action_batch)
        td_estimate_z = z.gather(1, z_action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values_x = torch.zeros(self.BATCH_SIZE, device=self.device)
        next_state_values_y = torch.zeros(self.BATCH_SIZE, device=self.device)
        next_state_values_z = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            # #DQN
            # ns_pitch, ns_yaw = self.target_net(non_final_next_states)
            # next_state_values_pitch[non_final_mask] = ns_pitch.max(1)[0]
            # next_state_values_yaw[non_final_mask] = ns_yaw.max(1)[0]

            #DDQN
            ns_target_x, ns_target_y, ns_target_z = self.target_net(non_final_next_states)
            ns_policy_x, ns_policy_y, ns_policy_z = self.policy_net(non_final_next_states)
            best_action_x = torch.argmax(ns_policy_x, axis=1).unsqueeze(0).t()
            best_action_y = torch.argmax(ns_policy_y, axis=1).unsqueeze(0).t()
            best_action_z = torch.argmax(ns_policy_z, axis=1).unsqueeze(0).t()

            next_state_values_x[non_final_mask] = ns_target_x.gather(1, best_action_x).squeeze() #Qt(S_t+1, argmax_a Qp(S_t+1,a))
            next_state_values_y[non_final_mask] = ns_target_y.gather(1, best_action_y).squeeze() #Qt(S_t+1, argmax_a Qp(S_t+1,a))
            next_state_values_z[non_final_mask] = ns_target_z.gather(1, best_action_z).squeeze() #Qt(S_t+1, argmax_a Qp(S_t+1,a))

        # Compute the expected Q values
        td_target_x = (next_state_values_x * self.GAMMA) + reward_batch  #R + gamma * target
        td_target_y = (next_state_values_y * self.GAMMA) + reward_batch  #R + gamma * target
        td_target_z = (next_state_values_z * self.GAMMA) + reward_batch  #R + gamma * target

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        x_loss = criterion(td_estimate_x, td_target_x.unsqueeze(1)) 
        y_loss = criterion(td_estimate_y, td_target_y.unsqueeze(1)) 
        z_loss = criterion(td_estimate_z, td_target_z.unsqueeze(1)) 

        loss = x_loss + y_loss + z_loss
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        return loss