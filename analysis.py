import numpy as np 
from matplotlib import pyplot as plt
import os 
import torch 

def episodic_returns(dir):
    train_returns = []
    train_x = []
    eval_returns = []
    eval_x = []
    for i, file_path in enumerate(sorted(os.listdir(dir))):
        # check if current file_path is a file
        file = os.path.join(dir, file_path)
        with open(file, 'rb') as f:
            r = np.load(f)
            if i % 10 == 0:
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

def remove_erm(file):
    checkpoint = torch.load(file, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    torch.save({
            'training_steps': checkpoint['training_steps'],
            'model_state_dict_policy': checkpoint['model_state_dict_policy'],
            'model_state_dict_target': checkpoint['model_state_dict_target'],
            'optimizer_state_dict': checkpoint['optimizer_state_dict'],
        }, file)
    print(file)
    return


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

directory = './trajectories/dqn/0'
file = './trajectories/dqn/0/episode_00510.npy'
target = './trajectories/targets/rope_center.npy'

episodic_returns(directory)
plot_trajectory(file, target)

