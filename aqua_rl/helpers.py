import numpy as np


def reward_calculation(detected, w, h):
    target = (w/2, h/2)
    max_dist = np.sqrt(np.square(w)+np.square(h))/2
    dist = np.sqrt(np.square(detected[0] - target[0]) + np.square(detected[1] - target[1]))/max_dist
    return 1 - dist
    
def action_mapping(idx, n):
    pitch = idx // n
    yaw = idx % n
    return pitch, yaw

def inverse_mapping(pitch_idx, yaw_idx, n):
    return int(pitch_idx * n + yaw_idx)
    
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