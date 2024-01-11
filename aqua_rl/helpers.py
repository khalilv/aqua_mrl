import numpy as np

def define_positive_template(img_size):
    t = np.zeros(img_size)
    half = int(img_size[1]/2)
    t[:,half-2:half+2] = 1
    return t.astype(np.uint8)

def define_negative_template(img_size):
    t = np.zeros(img_size)
    half = int(img_size[0]/2)
    t[half-2:half+2,:] = 1
    return t.astype(np.uint8)

def reward_calculation(seg_map, relative_depth, positive_template, negative_template, detection_threshold):
    # Calculate intersection and union
    pi = np.logical_and(seg_map, positive_template)
    pu = np.logical_or(seg_map, positive_template)
    positive_iou = np.sum(pi) / np.sum(pu)

    # Calculate intersection and union
    ni = np.logical_and(seg_map, negative_template)
    nu = np.logical_or(seg_map, negative_template)
    negative_iou = np.sum(ni) / np.sum(nu)

    if positive_iou > negative_iou:
        r = positive_iou
    else:
        r = negative_iou * -2

    if np.sum(seg_map) > detection_threshold:
        bonus = 0.05
    else:
        bonus = 0.0

    if np.abs(relative_depth) < 2:
        return r + bonus
    else:
        return -1.0