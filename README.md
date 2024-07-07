# Following scuba divers with an autonomous underwater vehicle 

This ROS2 package contains all code used in my thesis on *Following scuba divers with an autonomous underwater vehicle*. This code should be paired with the [Aquasim underwater robot simulator](https://www.independentrobotics.com/robot-simulator).

There are several relevant nodes:

- `autopilot.py` - Node to receive commands and publish actions to the simulator. It can receive either discretized action indices or raw commands.
- `detect.py` - Node to receive images from the simulator and detect and track divers in the image plane.
- `diver_controller.py` - Node to simulate diver movement.
- `dqn_controller.py` - Training procedure to use Double Deep Q-Networks (DDQN) to learn to follow a scuba diver through interaction with the environment.
- `evaluation.py` - Node to perform evaluations of the DDQN control strategy.
- `pid_controller.py` - Node to deploy traditional Proportional-Integral-Derivative (PID) controllers to follow a scuba diver.
- `resetter.py` - Script to reset the simulator and all active nodes. This is included to run simulations for long durations without manual intervention. Note that this script only kills all the relevant nodes. There should be a separate script running to automatically restart those nodes after they have been killed, something along the lines of:

```python
import time
import subprocess

while(1):
    returned_output = subprocess.run('pgrep ' + 'diver_controller', capture_output=True, shell=True)
    if returned_output.stdout.decode("utf-8")[:-1] =='':
        subprocess.run('ros2 run aqua_mrl diver_controller', shell=True)

    time.sleep(1)
```

- `hyperparams.py` - All hyperparameters used in experiments.
- `helpers.py` - Functions common to multiple nodes.
- `control/` - All code for PID and learning-based control strategies.
- `DeepLabv3/` - Segmentation module (not used in current work).
- `flow_magnitudes/` - Data from early experiments trying to estimate velocity using optical flow.
- `plots/` - Plots that explore the time delay in the simulation.
- `RAFT/` - Optical flow model (not used in current work).
- `unused_nodes/` - Nodes not currently used in scuba diver following experiments.
- `YOLOv7/`- Code used to detect and track scuba divers in the image plane.
- `imitation_learning_module/` - Code used to train a neural network to mimic expert behavior (not used in current work).
- `segmentation_module/` - Code used to train and evaluate DeepLabv3 for image segmentation (not used in current work).
- `analysis.py` - Functions used to analyze data and results.
- `best_ddqn.pt` - Best DDQN weights used in evaluation studies.
- `instructions.txt` - Basic instructions on how to set up the code and simulator (may be outdated).
- `reset_to_checkpoint` - Code to reset an experiment to a specific episode, to be used in case there are issues with the simulator that disrupt training.


The branch pipeline_tracking includes an extension of this work to autonomously inspect subsea cables or pipelines using visual servoing techniques. This branch is currently under development. 
