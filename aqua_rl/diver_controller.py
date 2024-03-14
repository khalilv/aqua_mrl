import rclpy
import numpy as np
from rclpy.node import Node
from aqua2_interfaces.msg import DiverCommand, AquaPose
from aqua_rl import hyperparams

class diver_controller(Node):

    def __init__(self):
        super().__init__('diver_controller')

        #hyperparams
        self.queue_size = hyperparams.queue_size_
        self.depth_range = hyperparams.depth_range_
        self.max_speed = hyperparams.diver_max_speed_

        #command publisher
        self.publisher = self.create_publisher(DiverCommand, 'diver_control', self.queue_size)
        self.pose_subscriber = self.create_subscription(
            AquaPose,
            hyperparams.diver_topic_name_,
            self.pose_callback,
            self.queue_size)
        
        #command
        self.cmd = DiverCommand()
        self.pose = None
        
        timer_period = 5  # seconds
        self.timer = self.create_timer(timer_period, self.publish_command)
        print('Initialized: diver controller')

    def publish_command(self):     
        if self.pose:

            #scale vector to current magnitude
            self.cmd.vx = 0.0
            self.cmd.vy = np.random.uniform(-1,1)
            self.cmd.vz = np.random.uniform(-1,1)

            speed = np.sqrt(np.square(self.cmd.vy) + np.square(self.cmd.vz))
            if speed > self.max_speed:
                self.cmd.vy = self.cmd.vy * self.max_speed/speed
                self.cmd.vz = self.cmd.vz * self.max_speed/speed
                
            if self.pose[1] > self.depth_range[0] and self.cmd.vy > 0:
                self.cmd.vy *= -1
            elif self.pose[1] < self.depth_range[1] and self.cmd.vy < 0:
                self.cmd.vy *= -1


            #publish
            self.publisher.publish(self.cmd)
            print('Publishing diver command')

            return 
    
    def pose_callback(self, pose):        
        self.pose = (pose.x, pose.y, pose.z)
        return 
    
def main(args=None):
    rclpy.init(args=args)

    publisher = diver_controller()

    rclpy.spin(publisher)

    publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()