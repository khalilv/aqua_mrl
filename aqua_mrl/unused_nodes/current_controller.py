import rclpy
import numpy as np
from rclpy.node import Node
from aqua2_interfaces.msg import UnderwaterAdversaryCommand
from aqua_mrl import hyperparams
from std_srvs.srv import SetBool
from std_msgs.msg import UInt8MultiArray

class current_controller(Node):

    def __init__(self):
        super().__init__('current_controller')

        #hyperparams
        self.queue_size = hyperparams.queue_size_
        self.xlim = hyperparams.adv_x_limit_
        self.ylim = hyperparams.adv_y_limit_
        self.zlim = hyperparams.adv_z_limit_
        self.x_action_space = hyperparams.adv_x_action_space_
        self.y_action_space = hyperparams.adv_y_action_space_
        self.z_action_space = hyperparams.adv_z_action_space_

        #command publisher
        self.publisher = self.create_publisher(UnderwaterAdversaryCommand, hyperparams.adv_unity_topic_name_, self.queue_size)
        self.start_stop_service = self.create_service(
            SetBool,
            hyperparams.adv_start_stop_,
            self.start_stop_callback)
        self.action_subscriber = self.create_subscription(
            UInt8MultiArray,
            hyperparams.adv_command_topic_name_,
            self.action_callback,
            self.queue_size)
        
        #command
        self.cmd = UnderwaterAdversaryCommand()

        self.current_x_values = np.linspace(-self.xlim, self.xlim, self.x_action_space)
        self.current_y_values = np.linspace(-self.ylim, self.ylim, self.y_action_space)
        self.current_z_values = np.linspace(-self.zlim, self.zlim, self.z_action_space)

        #flag to start/stop publishing
        self.publish_flag = False

        print('Initialized: current controller')

    def action_callback(self, action):
        if self.publish_flag:
            action = np.array(action.data)
            self.cmd.current_x =  self.current_x_values[action[0]]
            self.cmd.current_y =  self.current_y_values[action[1]]
            self.cmd.current_z =  self.current_z_values[action[2]]
            #publish
            self.publisher.publish(self.cmd)
        return 
    
    def start_stop_callback(self, request, response):
        if request.data:
            self.publish_flag = True
            response.message = 'Current controller started'
        else:
            self.publish_flag = False
            self.cmd.current_x = 0.0
            self.cmd.current_y = 0.0
            self.cmd.current_z = 0.0
            self.publisher.publish(self.cmd)
            response.message = 'Current controller stopped and reset'
        response.success = True
        return response

def main(args=None):
    rclpy.init(args=args)

    publisher = current_controller()

    rclpy.spin(publisher)

    publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()