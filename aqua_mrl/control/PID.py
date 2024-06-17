import numpy as np 
import time

class AnglePID:
    def __init__(self, target = 0.0, command_range = [-1.0,1.0],
                 gains = [0.0,0.0,0.0], reverse = False):
        self.target = self.convert_to_0_360(target)
        self.command_range = command_range
        self.last_error = None
        self.last_time = None
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

        #p-control
        command = (self.Kp * error)
        
        current_time = time.time()
        #add d-control and i-control
        if self.last_error is not None:
            dt = current_time - self.last_time
            error_dt = (error - self.last_error)/dt
            self.accumulator += error * dt
            command = command + (self.Ki * self.accumulator) + (self.Kd * error_dt) 

        self.last_error = error
        self.last_time = current_time
        
        #clip command to min/max values
        return np.clip(command, self.command_range[0], self.command_range[1])

    def convert_to_0_360(self, measurement):
        return (((measurement % 360) + 360) % 360)
    
    def calculate_angle_error(self, measurement): 
        #signed difference between two angles in [0, 360]. returns value in [-180,180]
        return (self.target - measurement + 540) % 360 - 180


class PID:
    def __init__(self, target = 0.0, command_range = [-1.0,1.0],
                gains = [0.0,0.0,0.0], reverse = False, normalization_factor = 1):
        self.target = target
        self.command_range = command_range
        self.Kp = gains[0]
        self.Ki = gains[1]
        self.Kd = gains[2]
        self.last_error = None
        self.last_time = None
        self.accumulator = 0.0
        self.reverse = reverse
        self.n_factor = normalization_factor
        

    def control(self, measurement):
        error = self.target - measurement
        
        # multiply by -1 if [- moves counter clockwise, + moves clockwise]
        if self.reverse:
            error = error * -1
        
        #normalize error to [-1,1]
        error = error / self.n_factor

        #p-control
        command = (self.Kp * error)
        
        current_time = time.time()
        #add d-control and i-control
        if self.last_error is not None:
            dt = current_time - self.last_time
            error_dt = (error - self.last_error)/dt
            self.accumulator += error * dt
            command = command + (self.Ki * self.accumulator) + (self.Kd * error_dt) 

        self.last_error = error
        self.last_time = current_time
        
        #clip command to min/max values
        return np.clip(command, self.command_range[0], self.command_range[1])