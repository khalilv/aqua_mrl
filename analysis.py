import numpy as np 
from matplotlib import pyplot as plt
import os 

def episodic_returns(dir):
    returns = []
    for file_path in sorted(os.listdir(dir)):
        # check if current file_path is a file
        file = os.path.join(dir, file_path)
        with open(file, 'rb') as f:
            r = np.load(f)
            returns.append(np.sum(r))
    plt.plot(returns)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Episodic Rewards')
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

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(xt, zt, yt, 'green', label='Target')
    ax.plot3D(xl, zl, yl, 'yellow', label='Trajectory')

    plt.legend()
    plt.show()

directory = './trajectories/dqn/0'
file = './trajectories/dqn/2/episode_00074.npy'
target = './trajectories/targets/rope_center.npy'

episodic_returns(directory)
plot_trajectory(file, target)