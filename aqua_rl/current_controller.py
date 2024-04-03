import rclpy
from rclpy.node import Node
from aqua2_interfaces.msg import UnderwaterAdversaryCommand
from aqua_rl import hyperparams
from aqua_rl.helpers import adv_action_mapping
from std_srvs.srv import SetBool
from std_msgs.msg import UInt8

class current_controller(Node):

    def __init__(self):
        super().__init__('current_controller')

        #hyperparams
        self.queue_size = hyperparams.queue_size_
        self.adv_madnitude = hyperparams.adv_magnitude_

        #command publisher
        self.publisher = self.create_publisher(UnderwaterAdversaryCommand, hyperparams.adv_unity_topic_name_, self.queue_size)
        self.start_stop_service = self.create_service(
            SetBool,
            hyperparams.adv_start_stop_,
            self.start_stop_callback)
        self.action_subscriber = self.create_subscription(
            UInt8,
            hyperparams.adv_command_topic_name_,
            self.action_callback,
            self.queue_size)
        
        #command
        self.cmd = UnderwaterAdversaryCommand()
        
        #flag to start/stop publishing
        self.publish_flag = False

        print('Initialized: current controller')

    def action_callback(self, action):
        if self.publish_flag:
            action_idx = int(action.data)
            x,y,z = adv_action_mapping(action_idx)
            #scale vector to current magnitude
            self.cmd.current_x = self.adv_madnitude * x
            self.cmd.current_z = self.adv_madnitude * z
            self.cmd.current_y = self.adv_madnitude * y 
            #publish
            self.publisher.publish(self.cmd)
            print('Publishing adversary command')
        return 
    
    def start_stop_callback(self, request, response):
        if request.data:
            self.publish_flag = True
            response.message = 'Current controller started'
        else:
            self.publish_flag = False
            self.cmd.current_x = 0.0
            self.cmd.current_z = 0.0
            self.cmd.current_y = 0.0
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