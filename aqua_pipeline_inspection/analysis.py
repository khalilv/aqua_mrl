import numpy as np 
from matplotlib import pyplot as plt

experiment = 3
trajectory_file = '/usr/local/data/kvirji/AQUA/aqua_pipeline_inspection/aqua_pipeline_inspection/trajectories/pid_trajectory_{}.npy'.format(str(experiment))
save_path = '/usr/local/data/kvirji/AQUA/aqua_pipeline_inspection/aqua_pipeline_inspection/plots/pid_trajectory_{}.png'.format(str(experiment))
target_trajectory = '/usr/local/data/kvirji/AQUA/aqua_pipeline_inspection/aqua_pipeline_inspection/trajectories/pipeline_center.npy'

offset_x = 47.14558
offset_z = -19.43558
offset_y = -99.88829
target_depth = -10.0

with open(target_trajectory, 'rb') as f:
    pipeline_x = np.load(f) + offset_x
    pipeline_z = np.load(f) + offset_z

with open(trajectory_file, 'rb') as f:
    trajectory = np.load(f)

def get_z_targ(xl, xt, zt):
    z_target = []
    for x in xl[:]:
        ind = np.argwhere(xt >= x)[0][0]
        x1 = xt[ind - 1]
        z1 = zt[ind - 1]
        x2 = xt[ind]
        z2 = zt[ind]
        m = (z2 - z1) / (x2 - x1)
        b = z2 - m * x2
        z_target.append(m*x + b)
    return z_target

def mse(traj_y, traj_z, targ_y, targ_z):
    return np.square(np.subtract(traj_y,targ_y)).mean() + np.square(np.subtract(traj_z,targ_z)).mean()

xl = trajectory[:,0]
yl = trajectory[:,1]
zl = trajectory[:,2]

yt = np.zeros(len(xl)) + target_depth
zt = get_z_targ(xl, pipeline_x, pipeline_z)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(xl, zl, yl, 'blue', label='Trajectory')
ax.plot3D(xl, zt, yt, 'green', label='Target')
MSE = mse(yl,zl,yt,zt)
plt.title('MSE: ' + str(MSE))
plt.legend()
plt.show()
