#subscriber/publisher queue size 
queue_size_ = 5

#img size
img_size_ = (32,32)

#control hyperparams
roll_gains_ = [0.25, 0.0, 0.75]
history_size_ = 20
speed_ = 0.25
pitch_limit_ = 0.008
yaw_limit_ = 0.25
yaw_action_space_ = 3
pitch_action_space_ = 3

#end of episode hyperparams
empty_state_max_ = 10
depth_range_ = [-6, -14.5]
starting_line_ = -72
finish_line_ = 70

#reward hyperparams
target_depth_ = -10.0
goal_reached_reward_ = 0.0
goal_not_reached_reward_ = -10.0
detection_threshold_ = 5

#training hyperparams
load_erm_ = True
experiment_number_ = 18
train_for_ = 10

#eval hyperparams
eval_episode_ = 10
eval_for_ = 10

#dirl
dirl_weights_ = 'src/aqua_rl/imitation_learning_module/models/best.pt' #set to None if starting from random weights