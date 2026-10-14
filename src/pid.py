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
    """Apply a PID correction to a sequence of predictions (ORIGINAL, ACAUSAL).

    This is the form used in the original study, kept so its results can be
    reproduced. It corrects prediction i using the error at step i -- that is,
    using the very value the model is trying to predict. That information does
    not exist at forecast time, so this cannot be deployed. See
    `correct_predictions_causal` for the version that can.
    """
    pid = PIDController(kp, ki, kd, dt)
    corrected = []
    for pred, target in zip(predictions, targets):
        error = float(target) - float(pred)
        corrected.append(float(pred) + pid.step(error))
    return corrected


def correct_predictions_causal(predictions, targets, kp=0.5, ki=0.1, kd=0.05,
                               dt=1.0):
    """Apply a PID correction using only errors already observed.

    At step i the forecast for step i has not yet been verified, but the
    forecast for step i-1 has: hour i-1 has happened and its true value is
    known. So the correction applied to prediction i is driven by

        e[i-1] = target[i-1] - prediction[i-1]

    This is causal and deployable -- it is the standard way forecast bias is
    corrected in practice. It can only help to the extent that successive
    forecast errors are correlated; if the residuals are white noise there is
    nothing for the controller to latch onto. `residual_autocorrelation` below
    measures exactly that.
    """
    pid = PIDController(kp, ki, kd, dt)
    predictions = [float(p) for p in predictions]
    targets = [float(t) for t in targets]

    corrected = [predictions[0]]  # no prior error exists for the first step
    for i in range(1, len(predictions)):
        prev_error = targets[i - 1] - predictions[i - 1]
        corrected.append(predictions[i] + pid.step(prev_error))
    return corrected


def correct_predictions_seasonal(predictions, targets, k=0.30, period=24):
    """Correct a forecast using the error made one period earlier.

    NOT a PID. A PID acts on the most recent error; this acts on the error at
    the same hour of the previous day:

        corrected[i] = prediction[i] + k * (target[i-24] - prediction[i-24])

    It is causal -- hour i-24 has long since happened -- and it targets the one
    place where this model's residuals are not white (see
    `residual_autocorrelation`). For a single-term linear correction the
    variance-minimising k equals the autocorrelation at that lag, so k should
    be set from measured data rather than guessed, and fitted on a validation
    block rather than on the test block.
    """
    import numpy as np

    p = np.asarray(predictions, dtype=float).copy()
    t = np.asarray(targets, dtype=float)
    if len(p) <= period:
        return p
    err = t[:-period] - p[:-period]
    out = p.copy()
    out[period:] = p[period:] + k * err
    return out


def residual_autocorrelation(residuals, max_lag=6):
    """Autocorrelation of the forecast residuals, lag 1..max_lag.

    A causal correction exploits structure left in the residuals. Near-zero
    autocorrelation means the LSTM has already extracted the predictable part
    and no downstream controller can recover more.
    """
    import numpy as np

    r = np.asarray(residuals, dtype=float)
    r = r - r.mean()
    denom = float(np.dot(r, r))
    if denom == 0:
        return {lag: 0.0 for lag in range(1, max_lag + 1)}
    return {
        lag: float(np.dot(r[:-lag], r[lag:]) / denom)
        for lag in range(1, max_lag + 1)
    }
