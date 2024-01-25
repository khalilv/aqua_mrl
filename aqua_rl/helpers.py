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
        case 7:
            return (1,0,1)
        case 8:
            return (1,0,-1)
        case 9:
            return (-1,0,1)
        case 10:
            return (-1,0,-1)
        case 11:
            return (0,1,1)
        case 12:
            return (0,1,-1)
        case 13:
            return (0,-1,1)
        case 14:
            return (0,-1,-1)
        case 15:
            return (1,0,1)
        case 16:
            return (-1,0,1)
        case 17:
            return (1,0,-1)
        case 18:
            return (-1,0,-1)
        case _:
            raise IndexError
