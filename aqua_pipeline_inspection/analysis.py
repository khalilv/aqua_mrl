import numpy as np 
from matplotlib import pyplot as plt

experiment = 2
trajectory_file = '/usr/local/data/kvirji/AQUA/aqua_pipeline_inspection/aqua_pipeline_inspection/trajectories/pid_trajectory_{}.npy'.format(str(experiment))
save_path = '/usr/local/data/kvirji/AQUA/aqua_pipeline_inspection/aqua_pipeline_inspection/plots/pid_trajectory_{}.png'.format(str(experiment))
target_trajectory = '/usr/local/data/kvirji/AQUA/aqua_pipeline_inspection/aqua_pipeline_inspection/trajectories/pipeline_center.npy'

offset_x = 47.14558
offset_z = -19.43558

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

def mse(true,pred):
    return np.square(np.subtract(true,pred)).mean() 

xl = trajectory[:,0]
zl = trajectory[:,2]
target = get_z_targ(xl, pipeline_x, pipeline_z)

plt.plot(xl,target)
plt.plot(xl,zl)
plt.title('MSE: ' + str(mse(target, zl)))
plt.show()
