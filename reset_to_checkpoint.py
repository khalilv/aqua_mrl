import os 

experiment = None #experiment to reset
episode = None #episode to reset to 

print('Resetting experiment {} to episode {}'.format(str(experiment), str(episode)))
traj = './experiments/{}/trajectories/'.format(str(experiment))
weights = './experiments/{}/weights/'.format(str(experiment))
memory = './experiments/{}/erm/'.format(str(experiment))
for file_path in sorted(os.listdir(weights)):
    if os.path.isfile(os.path.join(weights, file_path)):
        if int(file_path[8:13]) > episode:
            os.remove(os.path.join(weights, file_path))
for file_path in sorted(os.listdir(traj)):
    if os.path.isfile(os.path.join(traj, file_path)):
        if int(file_path[8:13]) > episode:
            os.remove(os.path.join(traj, file_path))
for file_path in sorted(os.listdir(memory)):
    if os.path.isfile(os.path.join(memory, file_path)):
        if int(file_path[8:13]) > episode:
            os.remove(os.path.join(memory, file_path))