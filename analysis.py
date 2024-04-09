import numpy as np 
from matplotlib import pyplot as plt
import os 

def episodic_returns(exp):
    t_dir = './experiments/{}/trajectories/'.format(str(exp))
    e_dir = './experiments/{}/weights/'.format(str(exp))
    train_returns = []
    train_x = []
    eval_returns = []
    eval_x = []
    for i, file_path in enumerate(sorted(os.listdir(t_dir))):
        # check if current file_path is a file
        file = os.path.join(t_dir, file_path)
        if os.path.isfile(file):
            with open(file, 'rb') as f:
                r = np.load(f)
                if os.path.exists(os.path.join(e_dir, file_path.replace('npy', 'pt'))):
                    eval_returns.append(np.sum(r))
                    eval_x.append(i)
                else:
                    train_returns.append(np.sum(r))
                    train_x.append(i)

    plt.plot(train_x, train_returns, color='blue', label='Train')
    plt.plot(eval_x, eval_returns, color='yellow', label='Eval')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Episodic Rewards')
    plt.legend()
    plt.show()

def analyze_erm(root_path):
    import torch
    from aqua_rl import hyperparams
    from aqua_rl.control.DQN import ReplayMemory, Transition
    from matplotlib import pyplot as plt
    history_size = hyperparams.history_size_
    yaw_actions = np.linspace(-hyperparams.yaw_limit_, hyperparams.yaw_limit_, hyperparams.yaw_action_space_)
    pitch_actions = np.linspace(-hyperparams.pitch_limit_, hyperparams.pitch_limit_, hyperparams.pitch_action_space_)
    erm = ReplayMemory(10000)
    save_memory_path = os.path.join(root_path, 'erm')

    for file_path in sorted(os.listdir(save_memory_path), reverse=True):
        if os.path.isfile(os.path.join(save_memory_path, file_path)):
            if erm.__len__() < 10000:
                memory = torch.load(os.path.join(save_memory_path, file_path), map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                erm.memory += memory['memory'].memory

    for i in range(erm.__len__()):
        transition = erm.get(i)
        transition = Transition(*zip(*transition))
        state = torch.cat(transition.state).detach().cpu().numpy()
        try:
            next_state = torch.cat(transition.next_state).detach().cpu().numpy()
        except:
            continue
        pitch_action = torch.cat(transition.pitch_action).detach().cpu().numpy()[0][0]
        yaw_action = torch.cat(transition.yaw_action).detach().cpu().numpy()[0][0]
        pitch_reward = torch.cat(transition.pitch_reward).detach().cpu().numpy()        
        yaw_reward = torch.cat(transition.yaw_reward).detach().cpu().numpy()
        sy,sx = state[0][0:int(history_size*2):3], state[0][1:int(history_size*2)+1:3]
        nsy, nsx = next_state[0][0:int(history_size*2):3], next_state[0][1:int(history_size*2)+1:3]
        plt.figure(figsize=(30,30))
        plt.xlim([-1, 1])
        plt.ylim([1, -1])
        plt.plot(sx, sy, label='state')
        plt.plot(sx[0], sy[0], c='lightgreen', marker='o', markersize=5 )
        plt.plot(sx[-1], sy[-1], c='lightblue', marker='o', markersize=5 )
        plt.plot(nsx, nsy, label='next state')
        plt.plot(nsx[0], nsy[0], c='darkgreen', marker='o', markersize=5)
        plt.plot(nsx[-1], nsy[-1], c='darkblue', marker='o', markersize=5)
        plt.legend()
        title = "Action: "
        pitch = pitch_actions[int(pitch_action)]
        yaw = yaw_actions[int(yaw_action)]
        if pitch < 0:
            title += "(pitch {} up, ".format(np.abs(pitch))
        elif pitch > 0:
            title += "(pitch {} down, ".format(np.abs(pitch))
        elif pitch == 0.0:
            title += "(no pitch, "
        if yaw < 0:
            title += "yaw {} left) ".format(np.abs(yaw))
        elif yaw > 0:
            title += "yaw {} right) ".format(np.abs(yaw))
        elif yaw == 0.0:
            title += "no yaw) "
        title += 'pitch reward: {}, yaw reward: {}'.format(pitch_reward, yaw_reward)
        plt.suptitle(title)
        plt.show()

def analyze_adv_erm(root_path):
    # x+ = right 
    # Z+ = backwards
    # y+ = down
    from aqua_rl import hyperparams
    from aqua_rl.control.TD3 import ReplayBuffer
    from matplotlib import pyplot as plt
    history_size = hyperparams.history_size_
    erm = ReplayBuffer(int(history_size * 3), hyperparams.adv_action_space_, 10000)
    save_memory_path = os.path.join(root_path, 'erm/adv')
    for file_path in sorted(os.listdir(save_memory_path), reverse=True):
        if os.path.isfile(os.path.join(save_memory_path, file_path)):
            with open(os.path.join(save_memory_path, file_path), 'rb') as f:
                states = np.load(f)
                actions = np.load(f)
                next_states = np.load(f)
                rewards = np.load(f)
                not_dones = np.load(f)
                if erm.size + len(states) < 10000:
                    erm.add_batch(states,actions,next_states,rewards,not_dones,len(states))

    for i in range(erm.size):
        state, action, next_state, reward, not_done = erm.get(i)
        state = state.detach().cpu().numpy()
        next_state = next_state.detach().cpu().numpy()
        action = action.detach().cpu().numpy()
        reward = reward.detach().cpu().numpy()
        not_done = not_done.detach().cpu().numpy()
        sy,sx = state[0:int(history_size*2):3], state[1:int(history_size*2)+1:3]
        nsy, nsx = next_state[0:int(history_size*2):3], next_state[1:int(history_size*2)+1:3]
        plt.figure(figsize=(30,30))
        plt.xlim([-1, 1])
        plt.ylim([1, -1])
        plt.plot(sx, sy, label='state')
        plt.plot(sx[0], sy[0], c='lightgreen', marker='o', markersize=5 )
        plt.plot(sx[-1], sy[-1], c='lightblue', marker='o', markersize=5 )
        plt.plot(nsx, nsy, label='next state')
        plt.plot(nsx[0], nsy[0], c='darkgreen', marker='x', markersize=5)
        plt.plot(nsx[-1], nsy[-1], c='darkblue', marker='x', markersize=5)
        plt.legend()
        title = "Action: ("
        if action[0] < 0:
            title += '{} left '.format(action[0])
        else:
            title += '{} right '.format(action[0])
        if action[1] < 0:
            title += '{} forwards '.format(action[1])
        else:
            title += '{} backwards '.format(action[1])
        if action[2] < 0:
            title += '{} up '.format(action[2])
        else:
            title += '{} down '.format(action[2])

        title += ') reward: {}'.format(reward[0])
        plt.suptitle(title)
        plt.show()

def eval(exp):
    dir = './evaluations/{}_episode_best/'.format(exp)
    rewards = []
    for i, file_path in enumerate(sorted(os.listdir(dir))):
        # check if current file_path is a file
        file = os.path.join(dir, file_path)
        if os.path.isfile(file):
            with open(file, 'rb') as f:
                r = np.load(f)
                rewards.append(np.sum(r))
    rewards = np.array(rewards)
    print('Mean: {}, std: {}'.format(np.mean(rewards), np.std(rewards)))


experiment = 0
episode = 296
file = './experiments/{}/trajectories/episode_{}.npy'.format(str(experiment), str(episode).zfill(5))
target = './rope_center.npy'

# analyze_erm('/home/khalilv/Documents/aqua/aqua_rl/experiments/0')

analyze_adv_erm('/home/khalilv/Documents/aqua/aqua_rl/experiments/0')
