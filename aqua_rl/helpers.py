import numpy as np
from aqua_rl.control.DQN import ReplayMemory, Transition
import torch
import os 
from matplotlib import pyplot as plt

def define_template(img_size):
    t = np.zeros(img_size)
    half = int(img_size[1]/2)
    t[:,half-1:half+1] = 1
    return t.astype(np.uint8)

def reward_calculation(seg_map, relative_depth, template):
    # Calculate intersection and union
    intersection = np.logical_and(seg_map, template)
    union = np.logical_or(seg_map, template)
    iou = np.sum(intersection) / np.sum(union)

    if np.abs(relative_depth) < 2:
        return iou 
    else:
        return -0.1
    
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
    
    
def random_starting_position():
    starting_positions = [[70.0, -0.3],]
                        #   [32.0, 34.0],
                        #   [3.0, -22.0]]
    return starting_positions[np.random.randint(0,len(starting_positions))]
