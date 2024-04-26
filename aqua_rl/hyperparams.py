#subscriber/publisher queue size 
queue_size_ = 5

#img size
img_size_ = 416

#control hyperparams
diver_max_speed_ = 0.35

#autopilot hyperparams
use_autopilot_ = False
if use_autopilot_:
    #degrees
    pitch_limit_ = 10
    yaw_limit_ = 25
    pitch_gains_ = [3.5, 0.0, 1.0] #P,I,D
    yaw_gains_ = [3.5, 0.0, 1.0] #P,I,D
    max_speed_ = 1.0
    min_speed_= 0.25
else:
    #raw command
    pitch_limit_ = 0.075
    yaw_limit_ = 0.75
    max_speed_ = 1.0
    min_speed_= 0.25
    pitch_gains_ = [0.05, 0.0, 0.1] #P,I,D
    yaw_gains_ = [0.25, 0.0, 0.5] #P,I,D

#dqn hyperparams
history_size_ = 20
yaw_action_space_ = 7
pitch_action_space_ = 7
speed_action_space_ = 7
frame_skip_ = 1

#end of episode hyperparams
empty_state_max_ = 20
depth_range_ = [-6, -12]
train_duration_ = 1000

#reward hyperparams
location_sigma_ = 0.5
area_sigma_ = 0.025
target_area_ = 0.02

#adversary hyperparams
adv_x_action_space_ = 5
adv_y_action_space_ = 5
adv_z_action_space_ = 5
adv_x_limit_ = 2.0
adv_y_limit_ = 2.0
adv_z_limit_ = 2.0
#3 -> 0.3
#4 -> 0.5
#5 -> 0.0
#6 -> 0.75
#7 -> 1.0
#8 -> 2.0

#rarl hyperparams
switch_every_ = 200
switch_every_adv_ = 100

#training hyperparams
load_erm_ = True
experiment_number_ = 10
train_for_ = 10
pid_decay_start_ = 0
pid_decay_end_ = 0

#eval hyperparams
values_to_test_ = [1750,1500,1250,1000,850]
eval_prefix_ = 'bouyancy'
eval_episode_ = 700
eval_experiment_number_ = 8
eval_for_ = 5
eval_duration_ = 1000
using_hardware_topics_ = False

if using_hardware_topics_:
    command_topic_name_ = '/ramius/command'
    imu_topic_name_ = '/ramius/imu/filtered_data'
    depth_topic_name_ = '/ramius/depth'
    camera_topic_name_ = '/ramius/camera/left/image_raw/compressed'
else:
    command_topic_name_ = '/a13/command'
    imu_topic_name_ = '/aqua/pose'
    camera_topic_name_ = '/camera/left/image_raw/compressed'
    autopilot_pitch_gains_ = [0.5181, 0.0, 0.9] #P,I,D
    autopilot_yaw_gains_ = [0.5181, 0.0, 0.0] #P,I,D
    autopilot_roll_gains_ = [0.25, 0.0, 0.75] #P,I,D
    autopilot_command_ = '/autopilot/command'
    autopilot_start_stop_ = '/autopilot/start_stop'
    diver_start_stop_ = '/diver/start_stop'
    diver_pose_topic_name_ = '/diver/pose'
    diver_command_topic_name_ = 'diver_control'
    diver_reset_srv_name_ = '/diver/set_position'
    detection_topic_name_ = '/diver/coordinates'
    aqua_reset_srv_name_ = '/simulator/set_position'
    adv_unity_topic_name_ = '/adv/current'
    adv_command_topic_name_ = '/adv/command'
    adv_start_stop_ = '/adv/start_stop'
    bouyancy_srv_name_ = '/set_bouyancy'
    diver_seed_srv_name_ = '/diver/set_seed'
    diver_speed_srv_name_ = '/diver/set_speed'
    autopilot_yaw_limit_name_ = '/autopilot/set_yaw_limit'
    autopilot_pitch_limit_name_ = '/autopilot/set_pitch_limit'
