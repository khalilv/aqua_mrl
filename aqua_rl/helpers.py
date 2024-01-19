import numpy as np

def vertical_alignment_score(mask, mi):
    hw = mask.shape[1]/2
    column_indices = np.where(np.any(mask, axis=0))[0]
    var = np.var(column_indices)
    d = np.abs(np.mean(column_indices) - hw)/hw
    score = (1 - d*mi) / (1 + var)
    return score

def reward_calculation(seg_map, relative_depth, detection_threshold, mi):
    if np.sum(seg_map) > detection_threshold:
        r = vertical_alignment_score(seg_map, mi)
    else:
        r = -0.5
    
    if np.abs(relative_depth) < 2:
        d = 0.0
    else:
        d = -0.5
    
    return r + d
   
def random_starting_position():
    starting_positions = [[70.0, -0.3],
                          [60.0, -0.3],
                          [32.0, 31.0],
                          [-5.0, -25.0],
                          [-17.0, -16.0]]
    return starting_positions[np.random.randint(0,len(starting_positions))]

def analyze_erm(erm_path):
    import torch
    from aqua_rl.control.DQN import ReplayMemory, Transition
    from matplotlib import pyplot as plt
    pitch_actions = yaw_actions = [-1,0,1]
    history_size = 10
    target_depth = -10
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
        title = "Action: "
        action_idx = action[0][0]
        pitch = pitch_actions[int(action_idx/len(yaw_actions))]
        yaw = yaw_actions[action_idx % len(yaw_actions)]
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
        fig.suptitle(title, fontsize=30)
        plt.show()