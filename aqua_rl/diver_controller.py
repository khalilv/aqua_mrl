import rclpy
import numpy as np
from rclpy.node import Node
from aqua2_interfaces.msg import DiverCommand, AquaPose
from aqua_rl import hyperparams
from std_srvs.srv import SetBool
from ir_aquasim_interfaces.srv import SetPosition
from geometry_msgs.msg import Pose

class diver_controller(Node):

    def __init__(self):
        super().__init__('diver_controller')

        #hyperparams
        self.queue_size = hyperparams.queue_size_
        self.depth_range = hyperparams.depth_range_
        self.max_speed = hyperparams.diver_max_speed_

        #command publisher
        self.publisher = self.create_publisher(DiverCommand, hyperparams.diver_command_topic_name_, self.queue_size)
        self.pose_subscriber = self.create_subscription(
            AquaPose,
            hyperparams.diver_pose_topic_name_,
            self.diver_pose_callback,
            self.queue_size)
        self.start_stop_service = self.create_service(
            SetBool,
            hyperparams.diver_start_stop_,
            self.start_stop_callback)
        self.reset_diver_client = self.create_client(SetPosition, hyperparams.diver_reset_srv_name_)

        #command
        self.diver_cmd = DiverCommand()
        self.diver_pose = None
        
        #reset diver request
        self.reset_diver_req = SetPosition.Request()
        
        #starting diver position and orientation
        self.starting_pose = Pose()
        self.starting_pose.position.x = 62.5
        self.starting_pose.position.z = 0.3                               
        self.starting_pose.position.y = -10.0
        self.starting_pose.orientation.x = 0.4976952
        self.starting_pose.orientation.y = -0.5022942
        self.starting_pose.orientation.z = 0.4976952
        self.starting_pose.orientation.w = 0.5022942
        
        #flag to move diver
        self.move_diver = False

        timer_period = 5  # seconds
        self.timer = self.create_timer(timer_period, self.publish_diver_command)       
        print('Initialized: diver controller')

    def publish_diver_command(self):     
        if self.diver_pose is not None and self.move_diver:

            #scale vector to current magnitude
            self.diver_cmd.vx = np.random.uniform(hyperparams.speed_, hyperparams.speed_+0.05)
            self.diver_cmd.vy = np.random.uniform(-1,1)
            self.diver_cmd.vz = np.random.uniform(-1,1)

            speed = np.sqrt(np.square(self.diver_cmd.vy) + np.square(self.diver_cmd.vz))
            if speed > self.max_speed:
                self.diver_cmd.vy = self.diver_cmd.vy * self.max_speed/speed
                self.diver_cmd.vz = self.diver_cmd.vz * self.max_speed/speed
                
            if self.diver_pose[1] > self.depth_range[0] and self.diver_cmd.vy > 0:
                self.diver_cmd.vy *= -1
            elif self.diver_pose[1] < self.depth_range[1] and self.diver_cmd.vy < 0:
                self.diver_cmd.vy *= -1

            #publish
            self.publisher.publish(self.diver_cmd)
            return 
    
    def diver_pose_callback(self, pose):        
        self.diver_pose = [pose.x, pose.y, pose.z]
        return 
    
    def start_stop_callback(self, request, response):
        if request.data:
            self.move_diver = True
            response.message = 'Diver controller started'
        else:
            self.move_diver = False
            self.reset_diver_req.pose = self.starting_pose
            self.reset_diver_client.call_async(self.reset_diver_req)
            response.message = 'Diver controller stopped and reset'
        response.success = True
        return response
    
def main(args=None):
    rclpy.init(args=args)

    publisher = diver_controller()

    rclpy.spin(publisher)

    publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()