import numpy as np 
import os 
from matplotlib import pyplot as plt
from aqua_rl import hyperparams

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

def bouyancy_analysis(experiments):
    for exp in experiments:
        return_mean =[]
        return_std = []
        for b in hyperparams.values_to_test_:
            directory = '/usr/local/data/kvirji/AQUA/aqua_rl/evaluations/{}/bouyancy_{}/'.format(str(exp), str(b))
            if os.path.exists(directory):
                results = []
                for file in os.listdir(directory):
                    with open(os.path.join(directory, file), 'rb') as f:
                        r = np.load(f)
                        results.append(np.sum(r))
                return_mean.append(np.mean(results))
                return_std.append(np.std(results))
        print(return_mean, return_std)

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
        #plt.plot(train_x, train_returns, label='Train {}'.format(str(exp)))
        plt.plot(eval_x, eval_returns, label='Eval {}'.format(str(exp)))
    
    plt.xlabel('Episode')
    plt.ylabel('Total Return')
    plt.title('Episodic Return')
    plt.legend()
    plt.show()



# interdependency_study('/home/khalilv/Documents/aqua/aquasim_ws/interdependency/pitch_change', True)
# interdependency_study('/home/khalilv/Documents/aqua/aquasim_ws/interdependency/yaw_change', False)

bouyancy_analysis([5,8])
episodic_returns([5,8])
