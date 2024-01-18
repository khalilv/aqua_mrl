import numpy as np

def define_template(img_size):
    t = np.zeros(img_size)
    half = int(img_size[1]/2)
    t[:,half-1:half+1] = 1
    return t.astype(np.uint8)

def vertical_alignment_score(mask):
    # columns where 1s are present
    column_indices = np.where(np.any(mask, axis=0))[0]
    # variance
    var = np.var(column_indices)
    #lower variance = higher score
    score = 1 / (1 + var)
    return score

def reward_calculation(seg_map, relative_depth, detection_threshold):
    if np.sum(seg_map) > 10:
        r = vertical_alignment_score(seg_map)
    else:
        r = -0.25
    
    if np.abs(relative_depth) < 2:
        d = 0.0
    else:
        d = -0.25
    
    return r + d
   
def random_starting_position():
    starting_positions = [[70.0, -0.3]]
    #[32.0, 34.0]
    #[3.0, -22.0]
    return starting_positions[np.random.randint(0,len(starting_positions))]

def analyze_erm(erm_path):
    import torch
    from aqua_rl.control.DQN import ReplayMemory, Transition
    from matplotlib import pyplot as plt
    pitch_actions = [-0.008,0.0,0.008]
    yaw_actions = [-0.25,0.0,0.25]
    erm = ReplayMemory(10000)
    memory = torch.load(erm_path, map_location= torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    erm = memory['memory']
    while True:
        transition = erm.sample(1)
        transition = Transition(*zip(*transition))
        state = torch.cat(transition.state).detach().cpu().numpy()
        state_depths = torch.cat(transition.state_depths).detach().cpu().numpy()
        next_state = torch.cat(transition.next_state).detach().cpu().numpy()
        next_state_depths = torch.cat(transition.next_state_depths).detach().cpu().numpy()
        action = torch.cat(transition.action).detach().cpu().numpy()
        reward = torch.cat(transition.reward).detach().cpu().numpy()
        fig, axs = plt.subplots(2, 10, figsize=(30,30))
        for i in range(state.shape[1]):
            axs[0,i].imshow(state[0,i,:,:])
            axs[0,i].set_title(str(np.round(state_depths[0,i] + 10, decimals=3)))
        for i in range(next_state.shape[1]):
            axs[1,i].imshow(next_state[0,i,:,:])
            axs[1,i].set_title(str(np.round(next_state_depths[0,i] + 10, decimals=3)))
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
        
# def define_negative_template(img_size):
#     t = np.zeros(img_size)
#     half = int(img_size[0]/2)
#     t[half-2:half+2,:] = 1
#     return t.astype(np.uint8)

# def define_float_template(img_size):
#     t = np.zeros(img_size)
#     hw = int(img_size[1]/2)
#     hh = int(img_size[0]/2)
#     t[:,hw-1:hw+1] = 1
#     t[0:hh, hw-6:hw-1] = 0.25
#     t[0:hh, hw+1:hw+6] = 0.25
#     t[0:int(hh*2/3), hw-12:hw-6] = 0.15
#     t[0:int(hh*2/3), hw+6:hw+12] = 0.15
#     t[0:int(hh*2/4), :hw-12] = 0.1
#     t[0:int(hh*2/4), hw+12:] = 0.1
#     return t 

# def float_reward_calculation(seg_map, template, relative_depth=0):
#     if np.abs(relative_depth) < 2:
#         return np.sum(np.multiply(seg_map, template))/64
#     else:
#         return -0.1

# def reward_calculation(seg_map, relative_depth, positive_template, negative_template, detection_threshold):
#     # Calculate intersection and union
#     pi = np.logical_and(seg_map, positive_template)
#     pu = np.logical_or(seg_map, positive_template)
#     positive_iou = np.sum(pi) / np.sum(pu)

#     # Calculate intersection and union
#     ni = np.logical_and(seg_map, negative_template)
#     nu = np.logical_or(seg_map, negative_template)
#     negative_iou = np.sum(ni) / np.sum(nu)

#     if positive_iou > negative_iou:
#         r = positive_iou
#     else:
#         r = negative_iou * -2

#     if np.sum(seg_map) > detection_threshold:
#         bonus = 0.05
#     else:
#         bonus = 0.0

#     if np.abs(relative_depth) < 2:
#         return r + bonus
#     else:
#         return -0.25