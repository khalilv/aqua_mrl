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
        location_reward = np.exp(-distance**2 / (2 * location_sigma**2)) #reward based on 2D Gaussian distribution
        area_reward = np.exp(-(a-target_area)**2 / (2 * area_sigma**2)) #reward based on 2D Gaussian distribution
        reward = location_reward*area_reward
        reward = np.clip(reward, 0, 1) #clip reward to range [0, 1]
    else:
        reward = -1.0
    return reward


# reward_close = []
# reward_far = []
# reward_target = []

# for y in np.linspace(-1,1,416):
#     row_close = []
#     row_far = []
#     row_target = []
#     for x in np.linspace(-1,1,416):
#         row_close.append(reward_calculation(y,x, 0.05, 1.0,0.5, 0.025, 0.02))
#         row_far.append(reward_calculation(y,x, 0.001, 1.0,0.5, 0.025, 0.02))
#         row_target.append(reward_calculation(y,x, 0.02, 1.0,0.5, 0.025, 0.02))
#     reward_close.append(row_close)
#     reward_far.append(row_far)
#     reward_target.append(row_target)

# from matplotlib import pyplot as plt
 
# fig, axs = plt.subplots(1,3, figsize=(18, 5))
# axs[0].imshow(reward_close, extent=[-1, 1, 1, -1], vmin=0, vmax = 1)
# axs[1].imshow(reward_target, extent=[-1, 1, 1, -1], vmin=0, vmax=1)
# last = axs[2].imshow(reward_far, extent=[-1, 1, 1, -1], vmin=0, vmax=1)
# for i,a in enumerate([0.05,0.02,0.001]):
#     axs[i].set_xlim(-1, 1)
#     axs[i].set_ylim(1, -1)
#     axs[i].set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
#     axs[i].set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
#     axs[i].set_xlabel('Normalized x-coordinate')
#     axs[i].set_ylabel('Normalized y-coordinate')
#     axs[i].set_title('Normalized bounding box area: {}'.format(a))

# # Add a single colorbar for all subplots
# fig.colorbar(last, ax=axs, orientation='vertical', fraction=0.02, pad=0.04, label='Reward')

# # Display the plot
# plt.savefig('reward.png', dpi=600, bbox_inches='tight')

# plt.show()
# plt.colorbar()  # Add colorbar for reference

# # Set x and y axis limits
# plt.xlim(-1, 1)
# plt.ylim(1, -1)
# plt.xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
# plt.yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
# plt.clim(0,1)
# plt.xlabel('Normalized x-coordinate')
# plt.ylabel('Normalized y-coordinate')
# plt.title('Normalized bounding box area: 0.05')
# plt.savefig('reward_close.png', dpi=600, bbox_inches='tight')
