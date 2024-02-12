import numpy as np

def vertical_alignment_score(mask, mi):
    hw = mask.shape[1]/2
    column_indices = np.where(np.any(mask, axis=0))[0]
    var = np.var(column_indices)
    d = np.abs(np.mean(column_indices) - hw)/hw
    score = (1 - d*mi) / (1 + var)
    return score

def reward_calculation(seg_map, relative_depth, detection_threshold, mi):
    if np.sum(seg_map) > detection_threshold:
        r = vertical_alignment_score(seg_map, mi)
    else:
        r = -0.5
    
    if np.abs(relative_depth) < 2:
        d = 0.0
    else:
        d = -0.5
    
    return r + d
   
def random_starting_position():
    starting_positions = [[70.0, -0.3],
                          [60.0, -0.3],
                          [32.0, 31.0],
                          [-5.0, -25.0],
                          [-17.0, -16.0]]
    return starting_positions[np.random.randint(0,len(starting_positions))]

def adv_mapping(action):
    match action:
        case 0:
            return (0,0,0)
        case 1:
            return (-1,0,0)
        case 2:
            return (1,0,0)
        case 3:
            return (0,-1,0)
        case 4:
            return (0,1,0)
        case 5:
            return (0,0,-1)
        case 6:
            return (0,0,1)
        case _:
            raise IndexError


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