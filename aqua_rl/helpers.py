import numpy as np


def reward_calculation(yn, xn, detected, sharpness):
    if detected == 1.0:
        pitch_reward = sharpness/(np.abs(yn) + sharpness)
        yaw_reward = sharpness/(np.abs(xn) + sharpness)
    else:
        pitch_reward = -0.5
        yaw_reward = -0.5
    return pitch_reward, yaw_reward
    
def action_mapping(idx, n):
    pitch = idx // n
    yaw = idx % n
    return pitch, yaw

def inverse_mapping(pitch_idx, yaw_idx, n):
    return int(pitch_idx * n + yaw_idx)

def safe_region(steps, s, f):
    if steps < s:
        return 0.0
    elif steps >= s and steps < f:
        return (steps - s)/(f-s)
    else:
        return 1.0

def map_missing_detection(yn, xn):
    if yn < 0:
        if xn < 0:
            return -1, -1
        else:
            return -1, 1
    else:
        if xn < 0:
            return 1, -1
        else:
            return 1, 1

def normalize_coords(y,x,w,h):
    xn = (2 * x / w) - 1
    yn = (2 * y / h) - 1
    return yn, xn

def adv_action_mapping(idx):
    action_mapping = {
        0: (0, 0, 0),
        1: (0, 0, 1),
        2: (0, 1, 0),
        3: (0, 1, 1),
        4: (1, 0, 0),
        5: (1, 0, 1),
        6: (1, 1, 0),
        7: (1, 1, 1)
    }
    return action_mapping[idx]
    
def euler_from_quaternion(quaternion):
    """
    Converts quaternion (w in last place) to euler roll, pitch, yaw
    quaternion = [x, y, z, w]
    Bellow should be replaced when porting for ROS 2 Python tf_conversions is done.
    """
    x = quaternion.x
    y = quaternion.y
    z = quaternion.z
    w = quaternion.w

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw