import rclpy
import numpy as np
from rclpy.node import Node
from aqua_rl.control.PID import AnglePID
from aqua2_interfaces.msg import AquaPose, Command
from aqua_rl import hyperparams
from std_msgs.msg import UInt8MultiArray
from std_srvs.srv import SetBool
from ir_aquasim_interfaces.srv import SetPosition
from geometry_msgs.msg import Pose
from aqua2_interfaces.srv import SetFloat
from time import sleep

class autopilot(Node):
    def __init__(self):
        super().__init__('autopilot')
        self.queue_size = hyperparams.queue_size_
        self.use_autopilot = hyperparams.use_autopilot_
        self.max_speed = hyperparams.max_speed_
        self.min_speed = hyperparams.min_speed_
        self.roll_gains = hyperparams.autopilot_roll_gains_
        self.pitch_gains = hyperparams.autopilot_pitch_gains_
        self.yaw_gains = hyperparams.autopilot_yaw_gains_
        self.pitch_limit = hyperparams.pitch_limit_
        self.yaw_limit = hyperparams.yaw_limit_
        self.pitch_action_space = hyperparams.pitch_action_space_
        self.yaw_action_space = hyperparams.yaw_action_space_
        self.speed_action_space = hyperparams.speed_action_space_

        self.imu_subscriber = self.create_subscription(AquaPose, hyperparams.imu_topic_name_, self.imu_callback, self.queue_size)
        self.command_publisher = self.create_publisher(Command, hyperparams.command_topic_name_, self.queue_size)
        self.action_subscriber = self.create_subscription(
            UInt8MultiArray,
            hyperparams.autopilot_command_,
            self.action_callback,
            self.queue_size)
        self.start_stop_service = self.create_service(
            SetBool,
            hyperparams.autopilot_start_stop_,
            self.start_stop_callback)
        self.reset_client = self.create_client(SetPosition, hyperparams.aqua_reset_srv_name_)
        self.yaw_limit_service = self.create_service(
            SetFloat,
            hyperparams.autopilot_yaw_limit_name_,
            self.set_yaw_limit)
        self.pitch_limit_service = self.create_service(
            SetFloat,
            hyperparams.autopilot_pitch_limit_name_,
            self.set_pitch_limit)
        self.roll_angle_service = self.create_service(
            SetFloat,
            hyperparams.roll_angle_srv_name_,
            self.set_roll_angle)
        
        self.measured_roll_angle = None
        self.measured_pitch_angle = None
        self.measured_yaw_angle = None
        
        self.pitch_actions = np.linspace(-self.pitch_limit, self.pitch_limit, self.pitch_action_space)
        self.yaw_actions = np.linspace(-self.yaw_limit, self.yaw_limit, self.yaw_action_space)
        self.speed_actions = np.linspace(self.min_speed, self.max_speed, self.speed_action_space)

        self.roll_target = 0.0
        self.pitch_target = 0.0
        self.yaw_target = 90.0

        self.roll_pid = AnglePID(target = self.roll_target, gains = self.roll_gains, reverse=True)
        self.pitch_pid = AnglePID(target = self.pitch_target, gains = self.pitch_gains)
        self.yaw_pid = AnglePID(target = self.yaw_target, gains = self.yaw_gains)

        self.command = Command()
        self.command.pitch = 0.0
        self.command.yaw = 0.0
        self.command.roll = 0.0
        self.command.heave = 0.0
        self.command.speed = 0.0

        self.publish_actions = False

        self.pitch_action_to_execute = None
        self.yaw_action_to_execute = None
        self.speed_action_to_execute = None

        self.reset_req = SetPosition.Request()
        #aqua starting position and orientation 
        self.starting_pose = Pose()
        self.starting_pose.position.x = 70.0
        self.starting_pose.position.z = -0.3                               
        self.starting_pose.position.y = -10.0
        self.starting_pose.orientation.x = 0.0
        self.starting_pose.orientation.y = -0.7071068
        self.starting_pose.orientation.z = 0.0
        self.starting_pose.orientation.w = 0.7071068
        
        print('Initialized: autopilot')

    def imu_callback(self, imu):
        self.measured_roll_angle = self.calculate_roll(imu)
        self.measured_pitch_angle = self.calculate_pitch(imu)
        self.measured_yaw_angle = self.calculate_yaw(imu)
        
        if self.publish_actions:
            self.command.roll = self.roll_pid.control(self.measured_roll_angle)
            if self.use_autopilot:
                self.command.pitch = self.pitch_pid.control(self.measured_pitch_angle)
                self.command.yaw = self.yaw_pid.control(self.measured_yaw_angle)
            else:
                self.command.pitch = self.pitch_action_to_execute
                self.command.yaw = self.yaw_action_to_execute
                self.command.speed = self.speed_action_to_execute
            self.command_publisher.publish(self.command)
        
        return
    
    def action_callback(self, a):
        actions = np.array(a.data)
        pitch_action = self.pitch_actions[int(actions[0])]
        yaw_action = self.yaw_actions[int(actions[1])]
        speed_action = self.speed_actions[int(actions[2])]
        if self.use_autopilot:
            self.pitch_pid.target = self.measured_pitch_angle + pitch_action
            self.yaw_pid.target = self.measured_yaw_angle + yaw_action
        else:
            self.pitch_action_to_execute = pitch_action
            self.yaw_action_to_execute = yaw_action
            self.speed_action_to_execute = speed_action
        return

    def start_stop_callback(self, request, response):
        if request.data:
            self.publish_actions = True
            if self.use_autopilot:
                self.pitch_pid.target = 0.0
                self.yaw_pid.target = 90.0
            else:
                self.pitch_action_to_execute = 0.0
                self.yaw_action_to_execute = 0.0
                self.speed_action_to_execute = 0.0
            response.message = 'Autopilot started'
        else:
            self.publish_actions = False
            for _ in range(self.queue_size):
                self.command.pitch = 0.0
                self.command.yaw = 0.0
                self.command.roll = 0.0
                self.command.heave = 0.0
                self.command.speed = self.min_speed
                self.command_publisher.publish(self.command)
            sleep(2.5)
            self.reset_req.pose = self.starting_pose
            self.reset_client.call_async(self.reset_req)
            response.message = 'Autopilot stopped and reset'
        response.success = True
        return response
        
    def calculate_roll(self, imu):
        return imu.roll
    
    def calculate_pitch(self, imu):
        return imu.pitch
    
    def calculate_yaw(self, imu):
        return imu.yaw
    
    def set_yaw_limit(self, request, response):
        print('Setting yaw limit to {}'.format(request.value))
        self.yaw_limit = request.value
        self.yaw_actions = np.linspace(-self.yaw_limit, self.yaw_limit, self.yaw_action_space)        
        response.msg = 'Set yaw limit successfully'
        return response
    
    def set_pitch_limit(self, request, response):
        print('Setting pitch limit to {}'.format(request.value))
        self.pitch_limit = request.value
        self.pitch_actions = np.linspace(-self.pitch_limit, self.pitch_limit, self.pitch_action_space)
        response.msg = 'Set pitch limit successfully'
        return response
    
    def set_roll_angle(self, request, response):
        print('Setting target roll angle to {}'.format(request.value))
        self.roll_pid.target = request.value
        response.msg = 'Set roll angle successfully'
        return response


def main(args=None):
    rclpy.init(args=args)

    node = autopilot()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

    
