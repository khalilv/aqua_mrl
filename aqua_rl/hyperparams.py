#subscriber/publisher queue size 
queue_size_ = 5

#img size
img_size_ = (32,32)
display_original_ = False

#control hyperparams
roll_gains_ = [0.25, 0.0, 0.75] #P,I,D
pitch_gains_ = [0.005, 0.0, 0.175] #P,I,D
speed_ = 0.25
pitch_limit_ = 0.005
yaw_limit_ = 0.25

#dqn hyperparams
history_size_ = 10
yaw_action_space_ = 3
pitch_action_space_ = 3

#end of episode hyperparams
empty_state_max_ = 30
depth_range_ = [-6, -14.5]
starting_line_ = -72
finish_line_ = 70
max_duration_ = 3000

#reward hyperparams
target_depth_ = -10.0
goal_reached_reward_ = 0.0
goal_not_reached_reward_ = -1.0
detection_threshold_ = 5
roi_detection_threshold_ = 10
mean_importance_ = 0.25 #[0,1]

#training hyperparams
load_erm_ = True
experiment_number_ = 0
train_for_ = 5
frames_to_skip_ = 5

#eval hyperparams
eval_episode_ = -1
eval_for_ = 3

#adversary hyperparams
adv_action_space_ = 19
adv_magnitude_x_ = 0.2
adv_magnitude_y_ = 0.005
adv_magnitude_z_ = 0.2

#switch between adv and pro
switch_every_ = 600
switch_every_adv_ = 5

using_hardware_topics_ = False

if using_hardware_topics_:
    command_topic_name_ = '/ramius/command'
    imu_topic_name_ = '/ramius/imu/filtered_data'
    depth_topic_name_ = '/ramius/depth'
    camera_topic_name_ = '/ramius/camera/back/image_raw/compressed'
else:
    command_topic_name_ = '/a13/command'
    imu_topic_name_ = '/aqua/pose'
    depth_topic_name_ = '/aqua/depth'
    camera_topic_name_ = '/camera/back/image_raw/compressed'