import numpy as np

class PID_Controller:
    def __init__(self, Kp, Ki, Kd, integral_limit=1.0, output_limit=None):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.integral_limit, self.output_limit = integral_limit, output_limit
        self.integral = 0
    
    def update(self, pos_error, vel_error, dt):
        self.integral += pos_error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        
        output = self.Kp * pos_error + self.Ki * self.integral + self.Kd * vel_error
        
        if self.output_limit is not None:
            output = np.clip(output, -self.output_limit, self.output_limit)
        
        return output

def create_controllers(gain_dict):
    controllers = {}
    
    for name, gains in gain_dict.items():
        controllers[name] = PID_Controller(**gains)
    return controllers