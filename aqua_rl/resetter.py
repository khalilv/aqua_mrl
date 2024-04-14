from pynput.keyboard import Key, Controller
import subprocess
import time
import sys

is_adv = sys.argv[1]
is_adv = is_adv.lower() == 'true'

# This includes a set of steps to take to make sure that we reset almost everything in the simulator
# Starting with the nodes, we have to kill the nodes
# our controller
print('Timely reset initiated...')
nodes_list = ['supervisor','dqn_controller', 'dqn_adversary', 'default_server_', 'diver_controlle', 'autopilot', 'detect', 'current_control']
for i in range(len(nodes_list)):
    returned_output = subprocess.run('pgrep ' + nodes_list[i], capture_output=True, shell=True)
    if returned_output.stdout.decode("utf-8")[:-1] != '':
        subprocess.run('kill -9 ' + returned_output.stdout.decode("utf-8")[:-1], shell=True)

print('Resetting the simulator')
time.sleep(20)
# switch to the unity
command = 'wmctrl -a Coral Reef'
subprocess.run(command, shell=True)
time.sleep(2)

# re-playing the simulator
keyboard = Controller()

keyboard.press(Key.ctrl)
keyboard.press('p')

time.sleep(2)

keyboard.release('p')
keyboard.release(Key.ctrl)

time.sleep(2)

keyboard.press(Key.ctrl)
keyboard.press('p')

time.sleep(2)

keyboard.release('p')
keyboard.release(Key.ctrl)

time.sleep(2)

print('Calibrating')
subprocess.run('ros2 service call /a13/system/calibrate std_srvs/srv/Empty', shell=True)

time.sleep(15)
print('Switching to the swimming mode')
subprocess.run('ros2 service call /a13/system/set_mode aqua2_interfaces/srv/SetString "value: swimmode"',
               shell=True)

time.sleep(5)
if is_adv:
    print('Running the adversary..')
    subprocess.Popen('ros2 run aqua_rl dqn_adversary', shell=True)
else:
    print('Running the controller..')
    subprocess.Popen('ros2 run aqua_rl dqn_controller', shell=True)
