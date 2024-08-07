import numpy as np

def get_command(action, n_pitch_actions, n_yaw_actions):
    pitch_command = action % n_pitch_actions
    yaw_command = (action // n_pitch_actions) % n_yaw_actions
    return pitch_command, yaw_command

def safe_region(steps, s, f):
    if steps < s:
        return 0.0
    elif steps >= s and steps < f:
        return (steps - s)/(f-s)
    else:
        return 1.0
    
def normalize_coords(y,x,w,h):
    xn = (2 * x / w) - 1
    yn = (2 * y / h) - 1
    return yn, xn

def get_current(action, action_dim):
    base3_str = base_conversion(action, 3)
    base3_str = base3_str.zfill(action_dim)
    current = [int(digit, 3) for digit in base3_str]
    return current

def base_conversion(n, base):
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    result = ''
    while n > 0:
        result = digits[n % base] + result
        n //= base
    return result
    
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

def reward_calculation(yn, xn, a, detected, location_sigma, area_sigma, target_area):
    if detected == 1.0:
        distance = np.sqrt(xn**2 + yn**2) #distance from the origin
        location_reward = np.exp(-distance**2 / (2 * location_sigma**2)) #location reward
        area_reward = np.exp(-(a-target_area)**2 / (2 * area_sigma**2)) #area reward
        reward = location_reward*area_reward
        reward = np.clip(reward, 0, 1) #clip reward to range [0, 1]
    else:
        reward = -1.0
    return reward
