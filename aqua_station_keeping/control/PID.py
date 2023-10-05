import numpy as np 
import time

class AnglePID:
    def __init__(self, target = 0.0, command_range = [-1.0,1.0], error_range = [-180,180],
                 gains = [0.0,0.0,0.0], reverse = False):
        self.target = target
        self.command_range = command_range
        self.error_range = error_range
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
        if measurement > 180:
            error = self.target - (measurement  - 360)
        else:
            error = self.target - measurement
        
        # multiply by -1 if [- moves counter clockwise, + moves clockwise]
        if self.reverse:
            error = error * -1

        # normalize error in error range to [-1,1]
        #error = self.normalize(error)

        self.accumulator += error

        #p-control
        command = (self.Kp * error) + (self.Ki * self.accumulator)
        
        #add d-control
        if self.last_time is not None:
            dt = time.time() - self.last_time
            error_dt = (error - self.last_error)/dt
            command = command + (self.Kd * error_dt)

        self.last_error = error
        self.last_time = time.time()
        
        #clip command to min/max values
        return np.clip(command, self.command_range[0], self.command_range[1])

    def convert_to_0_360(self, measurement):
        return ((measurement % 360) + 360) % 360
    
    def normalize(self, error ):
        return 2* ((error - self.error_range[0]) / (self.error_range[1] - self.error_range[0])) - 1


class PID:
    def __init__(self, target = 0.0, command_range = [-1.0,1.0], error_range = [-10,10],
                gains = [0.0,0.0,0.0]):
        self.target = target
        self.command_range = command_range
        self.error_range = error_range
        self.Kp = gains[0]
        self.Ki = gains[1]
        self.Kd = gains[2]
        self.last_error = None
        self.last_time = None
        self.accumulator = 0.0
        

    def control(self, measurement):
        error = self.target - measurement

        self.accumulator += error
        
        # normalize error in error range to [-1,1]
        #error = self.normalize(error)

        #pi-control
        command = (self.Kp * error) + (self.Ki * self.accumulator)
        
        #add d-control
        if self.last_time is not None:
            dt = time.time() - self.last_time
            error_dt = (error - self.last_error)/dt
            command = command + (self.Kd * error_dt)

        self.last_error = error
        self.last_time = time.time()
        
        #clip command to min/max values
        return np.clip(command, self.command_range[0], self.command_range[1])

    def normalize(self, error):
        return 2* ((error - self.error_range[0]) / (self.error_range[1] - self.error_range[0])) - 1
