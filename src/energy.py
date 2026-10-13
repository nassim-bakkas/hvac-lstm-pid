"""
Thermal energy computation.

Fundamental relation used throughout the study:

    Q = m_dot * cp * |T - T_ref|

    Q      : thermal energy (J)
    m_dot  : mass flow rate (kg/s)
    cp     : specific heat capacity (J/kg.K)
    T_ref  : comfort setpoint (degC)

The setpoint of 22 degC is taken from the overlap of ASHRAE Standard 55
(20-26 degC) and ISO 7730 (22 degC).
"""

import numpy as np

# Defaults from the original study
REF_TEMP = 22.0  # degC, ASHRAE 55 / ISO 7730
MASS_FLOW = 1.0  # kg/s
CP_WATER = 4186.0  # J/(kg.K)


def calculate_energy(temp_real, temp_pred, ref_temp=REF_TEMP,
                     mass_flow=MASS_FLOW, cp=CP_WATER):
    """Return (Q_real, Q_pred) arrays in joules.

    Parameters
    ----------
    temp_real : array of measured temperatures (degC)
    temp_pred : array of LSTM-predicted temperatures (degC)
    ref_temp  : comfort setpoint (degC)
    mass_flow : mass flow rate (kg/s)
    cp        : specific heat capacity (J/kg.K)
    """
    delta_t_real = np.abs(np.asarray(temp_real, dtype=float) - ref_temp)
    delta_t_pred = np.abs(np.asarray(temp_pred, dtype=float) - ref_temp)

    q_real = mass_flow * cp * delta_t_real
    q_pred = mass_flow * cp * delta_t_pred
    return q_real, q_pred


def total_energy(q):
    """Sum an hourly energy array."""
    return float(np.sum(q))
