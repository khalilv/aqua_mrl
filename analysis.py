import numpy as np 
import os 
from matplotlib import pyplot as plt
import pandas as pd 
from scipy.interpolate import interp1d

def interdependency_study(dir, pitch_change):
    with open(os.path.join(dir, 'roll0.npy'), 'rb') as f:
        if pitch_change:
            angles_r0 = np.load(f)
            _ = np.load(f)
        else:
            _ = np.load(f)
            angles_r0 = np.load(f)
        times_r0 = np.load(f)
        times_r0 -= times_r0[0]
    with open(os.path.join(dir, 'roll10.npy'), 'rb') as f:
        if pitch_change:
            angles_r10 = np.load(f)
            _ = np.load(f)
        else:
            _ = np.load(f)
            angles_r10 = np.load(f)
        times_r10 = np.load(f)
        times_r10 -= times_r10[0]
    with open(os.path.join(dir, 'roll20.npy'), 'rb') as f:
        if pitch_change:
            angles_r20 = np.load(f)
            _ = np.load(f)
        else:
            _ = np.load(f)
            angles_r20 = np.load(f)
        times_r20 = np.load(f)
        times_r20 -= times_r20[0]
    with open(os.path.join(dir, 'rolln10.npy'), 'rb') as f:
        if pitch_change:
            angles_rn10 = np.load(f)
            _ = np.load(f)
        else:
            _ = np.load(f)
            angles_rn10 = np.load(f)
        times_rn10 = np.load(f)
        times_rn10 -= times_rn10[0]
    with open(os.path.join(dir, 'rolln20.npy'), 'rb') as f:
        if pitch_change:
            angles_rn20 = np.load(f)
            _ = np.load(f)
        else:
            _ = np.load(f)
            angles_rn20 = np.load(f)
        times_rn20 = np.load(f)
        times_rn20 -= times_rn20[0]
    
    if pitch_change:
        last_time = 17.5
    else:
        last_time = 17.5
    plt.plot(times_rn20[:np.argmax(times_rn20 > last_time)], angles_rn20[:np.argmax(times_rn20 > last_time)], label='Roll -20°', color='darkgreen')
    plt.plot(times_rn10[:np.argmax(times_rn10 > last_time)], angles_rn10[:np.argmax(times_rn10 > last_time)], label='Roll -10°', color='lime')
    plt.plot(times_r0[:np.argmax(times_r0 > last_time)], angles_r0[:np.argmax(times_r0 > last_time)], label='Roll 0°', color = 'orange')
    plt.plot(times_r10[:np.argmax(times_r10 > last_time)], angles_r10[:np.argmax(times_r10 > last_time)], label='Roll 10°', color = 'lightskyblue')
    plt.plot(times_r20[:np.argmax(times_r20 > last_time)], angles_r20[:np.argmax(times_r20 > last_time)], label='Roll 20°', color = 'darkblue')
    if pitch_change:
        plt.text(3, -60, 'Pitch rate = 0.0', fontsize=10)
        plt.text(11, -60, 'Pitch rate = 0.025', fontsize=10)
        plt.axvline(x=9, color='k', linestyle='--') 
        plt.ylabel('Yaw angle (degrees)')
    else:
        plt.text(3, -25, 'Yaw rate = 0.0', fontsize=10)
        plt.text(10, -25, 'Yaw rate = 0.25', fontsize=10)
        plt.ylim((-30,30))
        plt.axvline(x=9, color='k', linestyle='--') 
        plt.ylabel('Pitch Angle (degrees)')
    plt.xlabel('Time (s)')
    plt.legend()
    if pitch_change:
        plt.savefig('pitch_change.png', dpi = 1200)
    else:
        plt.savefig('yaw_change.png', dpi = 1200)
    plt.show()
    return

def density_analysis(experiments, names):
    colors = ['green', 'blue']
    ecolors = ['lightgreen', 'lightblue']
    for c, exp in enumerate(experiments):
        return_mean =[]
        return_std = []
        x = []
        values_to_test =  [3000.0,2500.0,2250.0,2000.0,1750.0,1500.0,1250.0,1000.0,850.0]
        for s in values_to_test:
            directory = '/usr/local/data/kvirji/AQUA/aqua_rl/evaluations/{}/bouyancy_{}/'.format(str(exp), str(s))
            if os.path.exists(directory):
                results = []
                for file in os.listdir(directory):
                    with open(os.path.join(directory, file), 'rb') as f:
                        r = np.load(f)
                        results.append(np.sum(r))
                return_mean.append(np.mean(results))
                return_std.append(np.std(results))
                x.append(s)
        plt.plot(x, return_mean, linestyle='-', color=colors[c], label=names[c])
        plt.fill_between(x, np.subtract(return_mean, return_std), np.add(return_mean, return_std), color = ecolors[c])
    plt.xlabel('Water density')
    plt.ylabel('Duration')
    plt.axvline(x=997.7, color='k', linestyle='--', label='Training density') 
    plt.legend()
    plt.show()

def speed_analysis(experiments):
    for exp in experiments:
        return_mean =[]
        return_std = []
        x = []
        values_to_test =  [0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0, 1.1, 1.2, 1.3]
        for s in values_to_test:
            directory = '/usr/local/data/kvirji/AQUA/aqua_rl/evaluations/{}/speed_{}/'.format(str(exp), str(s))
            if os.path.exists(directory):
                results = []
                for file in os.listdir(directory):
                    with open(os.path.join(directory, file), 'rb') as f:
                        r = np.load(f)
                        results.append(np.sum(r))
                return_mean.append(np.mean(results))
                return_std.append(np.std(results))
                x.append(s)
        plt.errorbar(x, return_mean, yerr=return_std, label=str(exp))
    plt.legend()
    plt.show()

def pitch_analysis(experiments, names):
    colors = ['green', 'blue']
    ecolors = ['lightgreen', 'lightblue']
    for c, exp in enumerate(experiments):
        return_mean =[]
        return_std = []
        x = []
        values_to_test =  [0.0025, 0.005, 0.0075, 0.0085, 0.015, 0.025, 0.065, 0.075, 0.085, 0.09, 0.1]
        for s in values_to_test:
            directory = '/usr/local/data/kvirji/AQUA/aqua_rl/evaluations/{}/pitch_{}/'.format(str(exp), str(s))
            if os.path.exists(directory):
                results = []
                for file in os.listdir(directory):
                    with open(os.path.join(directory, file), 'rb') as f:
                        r = np.load(f)
                        results.append(len(r))
                return_mean.append(np.mean(results))
                return_std.append(np.std(results))
                x.append(s)
        plt.plot(x, return_mean, linestyle='-', color=colors[c], label=names[c])
        plt.fill_between(x, np.subtract(return_mean, return_std), np.add(return_mean, return_std), color = ecolors[c])
    plt.xlabel('Pitch limit')
    plt.ylabel('Total reward')
    plt.axvline(x=0.05, color='k', linestyle='--', label='Training limit') 
    plt.legend()
    plt.show()


def episodic_returns(experiments):
    for exp in experiments:
        train_dir = '/usr/local/data/kvirji/AQUA/aqua_rl/experiments/{}/trajectories/'.format(str(exp))
        eval_dir = '/usr/local/data/kvirji/AQUA/aqua_rl/experiments/{}/weights/'.format(str(exp))
        train_returns = []
        eval_returns = []
        train_x = []
        eval_x = []
        for i,file_path in enumerate(sorted(os.listdir(train_dir))):
            file = os.path.join(train_dir, file_path)
            if os.path.isfile(file):
                with open(file, 'rb') as f:
                    r = np.load(f)
                    if os.path.exists(os.path.join(eval_dir, file_path.replace('npy', 'pt'))):
                        eval_returns.append(np.sum(r))
                        eval_x.append(i)
                    else:
                        train_returns.append(np.sum(r))
                        train_x.append(i)
        plt.plot(train_x, train_returns, label='Train {}'.format(str(exp)))
        plt.plot(eval_x, eval_returns, label='Eval {}'.format(str(exp)))
    
    plt.xlabel('Episode')
    plt.ylabel('Total Return')
    plt.title('Episodic Return')
    plt.legend()
    plt.show()

def location_analysis(evaluation_directory, location_bins=50, area_bins=50):
    rewards = []
    detected_locations = []
    for _,file_path in enumerate(sorted(os.listdir(evaluation_directory))):
        file = os.path.join(evaluation_directory, file_path)
        if os.path.isfile(file):
            with open(file, 'rb') as f:
                r = np.load(f)
                locations = np.load(f)
                rewards.append(np.sum(r))
                dl = locations[locations[:, 3] == 1]
                detected_locations.append(dl)
    detected_locations = np.vstack(detected_locations)
    heatmap, _, _ = np.histogram2d(detected_locations[:, 1], detected_locations[:, 0], bins=location_bins, range=[[-1,1],[-1,1]])
    extent = [-1, 1, -1, 1]
    plt.imshow(heatmap.T, extent=extent, origin='lower', interpolation='nearest', aspect='auto')
    c = plt.colorbar()
    plt.ylim(1, -1)
    plt.xticks([-1, -0.5, 0, 0.5, 1])
    plt.yticks([-1, -0.5, 0, 0.5, 1])
    plt.xlabel('Normalized x-coordinate')
    plt.ylabel('Normalized y-coordinate')
    plt.title('Diver Location Heatmap')
    c.set_label('Frames')
    plt.savefig('dqn_debris_location.png', dpi=600, bbox_inches='tight')
    plt.show()

    plt.hist(detected_locations[:,2], bins=area_bins, color='g', alpha=0.7)
    plt.title('Diver Size')
    plt.xlabel('Normalized bounding box area')
    plt.axvline(x=0.02, color='k', linestyle='--', label='Target area') 
    plt.legend()
    plt.xlim(0, 0.1)
    plt.ylabel('Frames')
    # plt.savefig('pid_baseline_area.png', dpi=600, bbox_inches='tight')

    plt.show()

def debris_analysis(dqn_directory, pid_directory, duration=False):
    N = 50
    x_max = 150000

    if duration:
        dqn_directory = os.path.join(dqn_directory, 'duration')
    else:
        dqn_directory = os.path.join(dqn_directory, 'rewards')

    # Define the filenames
    filenames = ['1_logs.csv', '2_logs.csv', '3_logs.csv']
    
    # Initialize lists to hold all x and y data
    all_x_data = []
    all_y_data = []
    
    # Loop through each file to read and store the data
    for filename in filenames:
        filepath = os.path.join(dqn_directory, filename)
        data = pd.read_csv(filepath)
        x = data.iloc[:, 1].values - 1473792
        y = data.iloc[:, 2].values 
        all_x_data.append(x)
        all_y_data.append(y)

    # Determine common x values
    x_min = min(min(x) for x in all_x_data)
    common_x_values = np.linspace(x_min, x_max, num=N) 

    # Interpolate y values at common x values
    interpolated_y_values = []
    for x, y in zip(all_x_data, all_y_data):
        f = interp1d(x, y, bounds_error=False, fill_value='extrapolate')
        interpolated_y_values.append(f(common_x_values))

    interpolated_y_values = np.array(interpolated_y_values)

    # Calculate mean and variance
    mean_y_values = np.mean(interpolated_y_values, axis=0)
    std_y_values = np.std(interpolated_y_values, axis=0)
    plt.plot(common_x_values, mean_y_values, label='DDQN', color='green')
    plt.fill_between(common_x_values, mean_y_values - std_y_values, mean_y_values + std_y_values, color='lightgreen')
    
    experiments = ['1','2','3']
    pid_y_values = []
    for exp in experiments:
        y = []
        filepath = os.path.join(pid_directory, exp)
        for file in sorted(os.listdir(filepath)):
            with open(os.path.join(filepath,file), 'rb') as f:
                r = np.load(f)
                if duration:
                    y.append(len(r))
                else:
                    y.append(np.sum(r))
        pid_y_values.append(y[0:N])
    
    pid_y_values = np.array(pid_y_values)
    pid_mean = np.mean(pid_y_values, axis=0)
    pid_std = np.std(pid_y_values, axis=0)
    plt.plot(common_x_values, pid_mean, label='PID', color='blue')
    plt.fill_between(common_x_values, pid_mean - pid_std, pid_mean + pid_std, color='lightblue')
    
    plt.legend()
    plt.xlabel('Interaction steps')
    if duration:
        plt.ylabel('Duration (frames)')
        plt.title('Duration')
        plt.savefig('density_duration.png', dpi=600, bbox_inches='tight')
    else:
        plt.title('Performance')
        plt.ylabel('Total Reward')
        plt.savefig('density_reward.png', dpi=600, bbox_inches='tight')
    plt.show()


def evaluation_analysis(dir):
    durations = []
    rewards = []
    detected_locations = []
    missed_detections = []
    for file in sorted(os.listdir(dir)):
        if os.path.isfile(os.path.join(dir,file)):
            with open(os.path.join(dir,file), 'rb') as f:
                r = np.load(f)
                locations = np.load(f)
                durations.append(len(r))
                rewards.append(np.sum(r))
                dl = locations[locations[:, 3] == 1]
                missed = len(locations[locations[:, 3] == 0])
                detected_locations.append(dl)
                missed_detections.append(missed)
    detected_locations = np.vstack(detected_locations)
    missed_detections = np.array(missed_detections)
    rewards = np.array(rewards)
    durations = np.array(durations)
    print('Duration: {} +- {}'.format(np.mean(durations), np.std(durations)))
    print('Reward: {} +- {}'.format(np.mean(rewards), np.std(rewards)))
    # print('Missed detections: {} +- {}'.format(np.mean(missed_detections), np.std(missed_detections)))
    print('Diver y-coordinates: {} +- {}'.format(np.mean(detected_locations[:, 0]), np.std(detected_locations[:,0])))
    print('Diver x-coordinates: {} +- {}'.format(np.mean(detected_locations[:, 1]), np.std(detected_locations[:,1])))

# interdependency_study('/home/khalilv/Documents/aqua/aquasim_ws/interdependency/pitch_change', True)
# interdependency_study('/home/khalilv/Documents/aqua/aquasim_ws/interdependency/yaw_change', False)

# density_analysis([5,8], ['Baseline','RARL'])
#location_analysis('/usr/local/data/kvirji/AQUA/aqua_rl/dqn_evaluations/halfdebris')
#debris_analysis('/usr/local/data/kvirji/AQUA/aqua_rl/experiments/density/results/', '/usr/local/data/kvirji/AQUA/aqua_rl/pid_evaluations/density', False)
#debris_analysis('/usr/local/data/kvirji/AQUA/aqua_rl/experiments/17/debris/', '/usr/local/data/kvirji/AQUA/aqua_rl/pid_evaluations/halfdebris', False)
evaluation_analysis('/usr/local/data/kvirji/AQUA/aqua_rl/pid_evaluations/baseline')
evaluation_analysis('/usr/local/data/kvirji/AQUA/aqua_rl/dqn_evaluations/baseline')