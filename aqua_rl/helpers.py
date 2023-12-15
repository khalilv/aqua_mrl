import numpy as np

def define_template(img_size, width):
    t = np.zeros(img_size)
    half = int(img_size[0]/2)
    hw = int(width/2)
    t[:,half-hw:half+hw] = 1
    return t.astype(np.uint8)

def reward_calculation(seg_map, relative_depth, template, alpha, beta):
    # Calculate intersection and union
    intersection = np.logical_and(seg_map, template)
    union = np.logical_or(seg_map, template)
    iou = np.sum(intersection) / np.sum(union)

    if np.abs(relative_depth) < 1:
        depth_reward = 0.5
    else: 
        depth_reward = -0.5

    return alpha*iou + beta*depth_reward
        
