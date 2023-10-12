import numpy as np 
import time

class AnglePID:
    def __init__(self, target = 0.0, command_range = [-1.0,1.0],
                 gains = [0.0,0.0,0.0], reverse = False):
        self.target = self.convert_to_0_360(target)
        self.command_range = command_range
        self.last_error = None
        self.Kp = gains[0]
        self.Ki = gains[1]
        self.Kd = gains[2]
        self.reverse = reverse
        self.accumulator = 0.0

    def control(self, measurement):
        #convert angle to 0-360 where + moves counter clockwise 
        measurement = self.convert_to_0_360(measurement)

        #define error [- moves clockwise, + moves counter clockwise ]
        error = self.calculate_angle_error(measurement)
        
        # multiply by -1 if [- moves counter clockwise, + moves clockwise]
        if self.reverse:
            error = error * -1

        #normalize error to [-1,1]
        error = error / 180

        self.accumulator += error

        #pi-control
        command = (self.Kp * error) + (self.Ki * self.accumulator)
        
        #add d-control
        if self.last_error is not None:
            error_dt = (error - self.last_error)
            command = command + (self.Kd * error_dt)

        self.last_error = error
        
        #clip command to min/max values
        return np.clip(command, self.command_range[0], self.command_range[1])

    def convert_to_0_360(self, measurement):
        return (((measurement % 360) + 360) % 360)
    
    def calculate_angle_error(self, measurement): 
        #signed difference between two angles in [0, 360]. returns value in [-180,180]
        return (self.target - measurement + 540) % 360 - 180


class PID:
    def __init__(self, target = 0.0, command_range = [-1.0,1.0],
                gains = [0.0,0.0,0.0]):
        self.target = target
        self.command_range = command_range
        self.Kp = gains[0]
        self.Ki = gains[1]
        self.Kd = gains[2]
        self.last_error = None
        self.accumulator = 0.0
        

    def control(self, measurement):
        error = self.target - measurement
       
        self.accumulator += error

        #pi-control
        command = (self.Kp * error) + (self.Ki * self.accumulator)
        
        #add d-control
        if self.last_error is not None:
            error_dt = (error - self.last_error)
            command = command + (self.Kd * error_dt)

        self.last_error = error
        
        #clip command to min/max values
        return np.clip(command, self.command_range[0], self.command_range[1])