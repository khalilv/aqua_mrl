import numpy as np

def define_template(img_size):
    t = np.zeros(img_size)
    half = int(img_size[0]/2)
    t[:,half-1:half+1] = 1
    return t.astype(np.uint8)

def reward_calculation(seg_map, relative_depth, template, thresh):
    # Calculate intersection and union
    intersection = np.logical_and(seg_map, template)
    union = np.logical_or(seg_map, template)
    iou = np.sum(intersection) / np.sum(union)
    
    #within depth range
    if np.abs(relative_depth) < 0.5:
        depth_reward = 0.2
    elif np.abs(relative_depth) < 1:
        depth_reward = 0.1
    elif np.abs(relative_depth) < 2:
        depth_reward = -0.1
    else: 
        depth_reward = -0.2

    #target is in image
    if np.sum(seg_map) >= thresh:
        r = iou + 0.2
    else:
        r = -0.2


    return r + depth_reward
        
