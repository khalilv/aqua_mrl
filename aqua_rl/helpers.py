import numpy as np


def reward_calculation(center, w, h, yaw_scale, pitch_scale):
    if center[0] == -1 and center[1] == -1:
        return -1, -1
    else:
        yaw_dist = np.abs(((w/2) - center[0])/(w/2))
        pitch_dist = np.abs(((h/2) - center[1])/(h/2))
        yaw_reward = yaw_scale/(yaw_dist + yaw_scale)
        pitch_reward = pitch_scale/(pitch_dist + pitch_scale)
        return pitch_reward, yaw_reward
    
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