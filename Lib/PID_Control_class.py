# PID Controller Class
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, error, dt):
        # Proportional term
        proportional = self.kp * error

        # Integral term
        self.integral += error * dt
        integral = self.ki * self.integral

        # Derivative term
        derivative = self.kd * (error - self.prev_error) / dt if dt > 0 else 0

        # Save current error for the next step
        self.prev_error = error

        # Total output
        return proportional + integral + derivative