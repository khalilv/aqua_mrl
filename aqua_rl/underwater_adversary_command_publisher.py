import rclpy
import numpy as np
from pynput import keyboard
from rclpy.node import Node
from aqua2_interfaces.msg import UnderwaterAdversaryCommand
from aqua_rl import hyperparams
from aqua_rl.helpers import adv_mapping

class underwater_adversary_command_publisher(Node):

    def __init__(self):
        super().__init__('underwater_adversary_command_publisher')

        #hyperparams
        self.queue_size = hyperparams.queue_size_
        self.adv_action_space = hyperparams.yaw_action_space_
        self.adv_madnitude_x = hyperparams.adv_magnitude_x_
        self.adv_madnitude_y = hyperparams.adv_magnitude_y_
        self.adv_madnitude_z = hyperparams.adv_magnitude_z_

        #command publisher
        self.publisher = self.create_publisher(UnderwaterAdversaryCommand, 'adv_command', self.queue_size)
        
        #command
        self.cmd = UnderwaterAdversaryCommand()
        
        #manual control with keys
        self.listener = keyboard.Listener(on_press=self.publish_command)
        self.listener.start()
        print('Initialized: underwater adversary')

    def publish_command(self, key):
        try:
            action = int(key.char)
            x,y,z = adv_mapping(action)
            
            #scale vector to current magnitude
            self.cmd.current_x = self.adv_madnitude_x * x
            self.cmd.current_z = self.adv_madnitude_z * z
            self.cmd.current_y = self.adv_madnitude_y * y 
            
            #publish
            self.publisher.publish(self.cmd)
            print('Publishing adversary command')

        except IndexError:
            print('Index out of range index for adversary action')
        except:
            print('Non-numeric key pressed')
        return 

def main(args=None):
    rclpy.init(args=args)

    publisher = underwater_adversary_command_publisher()

    rclpy.spin(publisher)

    publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()