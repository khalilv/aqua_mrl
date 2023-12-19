import os 

experiment = 0
episode = 20
traj = './experiments/{}/trajectories/'.format(str(experiment))
weights = './experiments/{}/weights/'.format(str(experiment))
for file_path in sorted(os.listdir(weights)):
    if int(file_path[8:13]) > episode:
        os.remove(os.path.join(weights, file_path))
for file_path in sorted(os.listdir(traj)):
    if int(file_path[8:13]) > episode:
        os.remove(os.path.join(traj, file_path))