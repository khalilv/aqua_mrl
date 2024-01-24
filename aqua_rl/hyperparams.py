#subscriber/publisher queue size 
queue_size_ = 5

#img size
img_size_ = (32,32)

#control hyperparams
roll_gains_ = [0.25, 0.0, 0.75]
history_size_ = 10
speed_ = 0.25
pitch_limit_ = 0.005
yaw_limit_ = 0.25
yaw_action_space_ = 3
pitch_action_space_ = 3

#end of episode hyperparams
empty_state_max_ = 30
depth_range_ = [-6, -14.5]
starting_line_ = -72
finish_line_ = 70
max_duration_ = 5000

#reward hyperparams
target_depth_ = -10.0
goal_reached_reward_ = 0.0
goal_not_reached_reward_ = -1.0
detection_threshold_ = 5
roi_detection_threshold_ = 10
mean_importance_ = 0.25 #[0,1]

#training hyperparams
load_erm_ = True
experiment_number_ = 30
train_for_ = 5
frames_to_skip_ = 5

#eval hyperparams
eval_episode_ = 356
eval_for_ = 3

#adversary hyperparams
adv_action_space_ = 11
adv_magnitude_x_ = 0.25
adv_magnitude_y_ = 0.05
adv_magnitude_z_ = 0.25

#switch between adv and pro
switch_every_ = 10
