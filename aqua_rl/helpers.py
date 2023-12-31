import numpy as np

def define_template(img_size):
    t = np.zeros(img_size)
    half = int(img_size[0]/2)
    t[:,half-1:half+1] = 1
    return t.astype(np.uint8)

def reward_calculation(seg_map, relative_depth, template):
    # Calculate intersection and union
    intersection = np.logical_and(seg_map, template)
    union = np.logical_or(seg_map, template)
    iou = np.sum(intersection) / np.sum(union)

    if np.abs(relative_depth) < 2:
        r = iou - 0.02
    else: 
        r = -0.5
    return r
        
