#subscriber/publisher queue size 
queue_size_ = 5

#img size
img_size_ = 416

#control hyperparams
pitch_angle_limit_ = 10
yaw_angle_limit_ = 25
diver_max_speed_ = 0.35

#dqn hyperparams
history_size_ = 20
yaw_action_space_ = 5
pitch_action_space_ = 5
frame_skip_ = 5

#end of episode hyperparams
empty_state_max_ = 20
depth_range_ = [-6, -12]
train_duration_ = 2000

#reward hyperparams
sharpness_ = 0.1

#training hyperparams
load_erm_ = True
experiment_number_ = 0
train_for_ = 10
pid_decay_start_ = 25000
pid_decay_end_ = 250000

#eval hyperparams
eval_episode_ = -1
eval_for_ = 10
eval_duration_ = 2000

diver_topic_name_ = '/diver/pose'
using_hardware_topics_ = False

if using_hardware_topics_:
    command_topic_name_ = '/ramius/command'
    imu_topic_name_ = '/ramius/imu/filtered_data'
    depth_topic_name_ = '/ramius/depth'
    camera_topic_name_ = '/ramius/camera/left/image_raw/compressed'
    speed_ = 0.6
else:
    command_topic_name_ = '/a13/command'
    imu_topic_name_ = '/aqua/pose'
    camera_topic_name_ = '/camera/left/image_raw/compressed'
    pitch_gains_ = [3.5, 0.0, 1.0] #P,I,D
    yaw_gains_ = [3.5, 0.0, 1.0] #P,I,D
    autopilot_pitch_gains_ = [0.5181, 0.0, 0.9] #P,I,D
    autopilot_yaw_gains_ = [0.5181, 0.0, 0.0] #P,I,D
    autopilot_roll_gains_ = [0.25, 0.0, 0.75] #P,I,D
    autopilot_command_ = '/autopilot/command'
    autopilot_start_stop_ = '/autopilot/start_stop'
    speed_ = 0.25
