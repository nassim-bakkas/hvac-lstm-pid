"""
PID controller.

In the original study the controller was applied as a correction term on
top of the LSTM forecast. Note the caveat in the README: a PID acts on a
*measured* error, so using it to post-correct a forecast only works online
if the error signal comes from an already-observed quantity.
"""


class PIDController:
    """Discrete PID controller.

    u(t) = Kp * e(t) + Ki * integral(e) + Kd * de/dt
    """

    def __init__(self, kp, ki, kd, dt=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def step(self, error):
        """Advance one timestep and return the control output."""
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output


def pid_controller(error, kp, ki, kd, integral=0.0, prev_error=0.0, dt=1.0):
    """Stateless form, kept to match the original annexe code."""
    integral += error * dt
    derivative = (error - prev_error) / dt
    output = kp * error + ki * integral + kd * derivative
    return output, integral, error


def correct_predictions(predictions, targets, kp=0.5, ki=0.1, kd=0.05, dt=1.0):
    """Apply a PID correction to a sequence of predictions.

    `targets` supplies the error signal. Offline this is the measured
    temperature; online it must be a quantity already observed at the time
    the correction is computed.
    """
    pid = PIDController(kp, ki, kd, dt)
    corrected = []
    for pred, target in zip(predictions, targets):
        error = float(target) - float(pred)
        corrected.append(float(pred) + pid.step(error))
    return corrected
