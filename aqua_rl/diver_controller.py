import rclpy
import numpy as np
from rclpy.node import Node
from aqua2_interfaces.msg import DiverCommand
from aqua_rl import hyperparams

class diver_controller(Node):

    def __init__(self):
        super().__init__('diver_controller')

        #hyperparams
        self.queue_size = hyperparams.queue_size_

        #command publisher
        self.publisher = self.create_publisher(DiverCommand, 'diver_control', self.queue_size)
        
        #command
        self.cmd = DiverCommand()
        
        timer_period = 5  # seconds
        self.timer = self.create_timer(timer_period, self.publish_command)
        print('Initialized: diver controller')

    def publish_command(self):     
            
        #scale vector to current magnitude
        self.cmd.vx = 0.4
        self.cmd.vy = np.random.uniform(-0.2,0.2)
        self.cmd.vz = np.random.uniform(-0.2,0.2)
            
        #publish
        self.publisher.publish(self.cmd)
        print('Publishing diver command')

        return 

def main(args=None):
    rclpy.init(args=args)

    publisher = diver_controller()

    rclpy.spin(publisher)

    publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()