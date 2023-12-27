import numpy as np 
from matplotlib import pyplot as plt
import os 
import torch 

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


experiment = 0
episode = 1
file = './experiments/{}/trajectories/episode_{}.npy'.format(str(experiment), str(episode).zfill(5))
target = './rope_center.npy'

episodic_returns(experiment)
depth_distribution(experiment)
plot_trajectory(file, target)



