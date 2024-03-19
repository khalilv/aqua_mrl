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

def depth_distribution(exp):
    t_dir = './experiments/{}/trajectories/'.format(str(exp))
    e_dir = './experiments/{}/weights/'.format(str(exp))
    train_depth_mean = []
    train_depth_std = []
    train_x = []
    eval_depth_mean = []
    eval_depth_std = []
    eval_x = []
    for i, file_path in enumerate(sorted(os.listdir(t_dir))):
        # check if current file_path is a file
        file = os.path.join(t_dir, file_path)
        if os.path.isfile(file):
            with open(file, 'rb') as f:
                _ = np.load(f)
                trajectory = np.load(f)
                depths = np.array(trajectory[:,1])
                if os.path.exists(os.path.join(e_dir, file_path.replace('npy', 'pt'))):
                    eval_depth_mean.append(np.mean(depths))
                    eval_depth_std.append(np.std(depths))
                    eval_x.append(i)
                else:
                    train_depth_mean.append(np.mean(depths))
                    train_depth_std.append(np.std(depths))
                    train_x.append(i)
                        
    plt.errorbar(x=train_x, y=train_depth_mean, yerr=train_depth_std, ecolor='red', fmt='.')
    plt.errorbar(x=eval_x, y=eval_depth_mean, yerr=eval_depth_std, ecolor='yellow', fmt='o')
    plt.show()

def duration_distribution(exp):
    t_dir = './experiments/{}/trajectories/'.format(str(exp))
    e_dir = './experiments/{}/weights/'.format(str(exp))
    train_duration = []
    train_x = []
    eval_duration = []
    eval_x = []
    for i, file_path in enumerate(sorted(os.listdir(t_dir))):
        # check if current file_path is a file
        file = os.path.join(t_dir, file_path)
        if os.path.isfile(file):
            with open(file, 'rb') as f:
                _ = np.load(f)
                trajectory = np.load(f)
                duration = trajectory.shape[0]
                if os.path.exists(os.path.join(e_dir, file_path.replace('npy', 'pt'))):
                    eval_duration.append(duration)
                    eval_x.append(i)
                else:
                    train_duration.append(duration)
                    train_x.append(i)
                        
    plt.plot(train_x, train_duration, color='blue', label='Train')
    plt.plot(eval_x, eval_duration, color='yellow', label='Eval')
    plt.xlabel('Episode')
    plt.ylabel('Timesteps')
    plt.title('Duration')
    plt.legend()
    plt.show()

def plot_trajectory(file, target):
    with open(file, 'rb') as f:
        _ = np.load(f)
        trajectory = np.load(f)
        xl = trajectory[:,0]
        yl = trajectory[:,1]
        zl = trajectory[:,2]

    with open(target, 'rb') as f:
        xt = np.load(f)
        yt = np.load(f)
        zt = np.load(f)

    ax = plt.axes(projection='3d')
    ax.plot3D(xt, zt, yt, 'green', label='Target')
    ax.plot3D(xl, zl, yl, 'yellow', label='Trajectory')

    plt.legend()
    plt.show()

def analyze_erm(erm_path):
    import torch
    from aqua_rl import hyperparams
    from aqua_rl.helpers import action_mapping
    from aqua_rl.control.DQN import ReplayMemory, Transition
    from matplotlib import pyplot as plt
    history_size = hyperparams.history_size_
    yaw_actions = np.linspace(-hyperparams.yaw_limit_, hyperparams.yaw_limit_, hyperparams.yaw_action_space_)
    pitch_actions = np.linspace(-hyperparams.pitch_limit_, hyperparams.pitch_limit_, hyperparams.pitch_action_space_)
    erm = ReplayMemory(10000)
    memory = torch.load(erm_path, map_location= torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    erm = memory['memory']
    while True:
        transition = erm.sample(1)
        transition = Transition(*zip(*transition))
        state = torch.cat(transition.state).detach().cpu().numpy()
        next_state = torch.cat(transition.next_state).detach().cpu().numpy()
        action = torch.cat(transition.action).detach().cpu().numpy()
        reward = torch.cat(transition.reward).detach().cpu().numpy()
        sx,sy = state[0][0:int(history_size*2):2], state[0][1:int(history_size*2)+1:2]
        nsx, nsy = next_state[0][0:int(history_size*2):2], next_state[0][1:int(history_size*2)+1:2]
        plt.figure(figsize=(30,30))
        plt.xlim([0, hyperparams.img_size_])
        plt.ylim([0, hyperparams.img_size_])
        plt.plot(sx, sy, label='state')
        plt.plot(sx[0], sy[0], c='g', marker='o', markersize=5 )
        plt.plot(sx[-1], sy[-1], c='b', marker='o', markersize=5 )
        plt.plot(nsx, nsy, label='next state')
        plt.plot(nsx[0], nsy[0], c='g', marker='o', markersize=5)
        plt.plot(nsx[-1], nsy[-1], c='b', marker='o', markersize=5)
        plt.legend()
        title = "Action: "
        action_idx = action[0][0]
        pitch_idx, yaw_idx = action_mapping(action_idx, hyperparams.yaw_action_space_)
        pitch = pitch_actions[pitch_idx]
        yaw = yaw_actions[yaw_idx]
        if pitch < 0:
            title += "(pitch up, "
        elif pitch > 0:
            title += "(pitch down, "
        elif pitch == 0.0:
            title += "(no pitch, "

        if yaw < 0:
            title += "yaw left)"
        elif yaw > 0:
            title += "yaw right) "
        elif yaw == 0.0:
            title += "no yaw) "

        title += 'Reward: {}'.format(reward)
        plt.suptitle(title)
        plt.show()

def analyze_adv_erm(erm_path):
    import torch
    from aqua_rl import hyperparams
    from aqua_rl.helpers import adv_mapping
    from aqua_rl.control.DQN import ReplayMemory, Transition
    from matplotlib import pyplot as plt
    history_size = hyperparams.history_size_
    target_depth = hyperparams.target_depth_
    erm = ReplayMemory(10000)
    memory = torch.load(erm_path, map_location= torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    erm = memory['memory']
    while True:
        transition = erm.sample(1)
        transition = Transition(*zip(*transition))
        state = torch.cat(transition.state).detach().cpu().numpy()
        state_depths = torch.cat(transition.state_depths).detach().cpu().numpy()
        state_actions = torch.cat(transition.state_actions).detach().cpu().numpy()
        next_state = torch.cat(transition.next_state).detach().cpu().numpy()
        next_state_depths = torch.cat(transition.next_state_depths).detach().cpu().numpy()
        next_state_actions = torch.cat(transition.next_state_actions).detach().cpu().numpy()
        action = torch.cat(transition.action).detach().cpu().numpy()
        reward = torch.cat(transition.reward).detach().cpu().numpy()
        fig, axs = plt.subplots(2, history_size, figsize=(30,30))
        for i in range(state.shape[1]):
            axs[0,i].imshow(state[0,i,:,:])
            axs[0,i].set_title("{}, {}".format(np.round(state_depths[0,i] - target_depth, decimals=3), state_actions[0,i]))
        for i in range(next_state.shape[1]):
            axs[1,i].imshow(next_state[0,i,:,:])
            axs[1,i].set_title("{}, {}".format(np.round(next_state_depths[0,i] - target_depth, decimals=3), next_state_actions[0,i]))
        action_idx = action[0][0]
        x, y, z = adv_mapping(action_idx)
        title = "Action (x: {}, y: {}, z: {})".format(x,y,z)
        title += 'Reward: {}'.format(reward)
        fig.suptitle(title, fontsize=30)
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

analyze_erm('/usr/local/data/kvirji/AQUA/aquasim_ws/pid_expert.pt')

