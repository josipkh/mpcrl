import numpy as np
import gymnasium as gym
import casadi as ca
from scipy.signal import cont2discrete
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Any
import matplotlib.pyplot as plt
from utils import contains_symbolics, cont2discrete_symbolic


@dataclass(kw_only=True)
class VehicleParams:
    cf:     float | ca.SX = 72705.0                 # [N/rad]   front cornering stiffness (for one tire)
    cr:     float | ca.SX = 72705.0                 # [N/rad]   rear cornering stiffness (for one tire)
    m:      float | ca.SX = 1600.0                  # [kg]      vehicle mass
    vx:     float | ca.SX = 60 / 3.6                # [m/s]     longitudinal vehicle speed
    lf:     float | ca.SX = 1.311                   # [m]       distance from the center of gravity to the front axle
    lr:     float | ca.SX = 1.311                   # [m]       distance from the center of gravity to the rear axle
    iz:     float | ca.SX = 2394.0                  # [kg*m^2]  vehicle moment of inertia
    isw:    float | ca.SX = 13                      # [-]       steering ratio
    sw_max: float | ca.SX = float(np.deg2rad(90))   # [rad] maximum steering wheel angle (wheel angle = steering wheel angle / steering ratio)


def get_A_cont(
    vehicle_params: dict[str, float | ca.SX] | None = VehicleParams()
) -> np.ndarray | ca.SX:
    cf = vehicle_params.cf
    cr = vehicle_params.cr
    m = vehicle_params.m
    vx = vehicle_params.vx
    lf = vehicle_params.lf
    lr = vehicle_params.lr
    iz = vehicle_params.iz

    row_1 = (0, 1, 0, 0)
    row_2 = (0,-(2*cf+2*cr)/(m*vx), (2*cf+2*cr)/m, -(2*cf*lf-2*cr*lr)/(m*vx))
    row_3 = (0, 0, 0, 1)
    row_4 = (0, -(2*cf*lf-2*cr*lr)/(iz*vx), (2*cf*lf-2*cr*lr)/iz, -(2*cf*lf**2+2*cr*lr**2)/(iz*vx))
    
    if contains_symbolics(vehicle_params):
        return ca.vertcat(
            ca.horzcat(*row_1),
            ca.horzcat(*row_2),
            ca.horzcat(*row_3),
            ca.horzcat(*row_4)
        )
    else:
        return np.array([row_1, row_2, row_3, row_4])
    

def get_B_steer_cont(
    vehicle_params: dict[str, float | ca.SX] | None = VehicleParams()
) -> np.ndarray | ca.SX:
    cf = vehicle_params.cf
    m = vehicle_params.m
    lf = vehicle_params.lf
    iz = vehicle_params.iz

    if any(isinstance(i, ca.SX) for i in [cf, m, lf, iz]):
        return ca.vertcat(
            0,
            2*cf/m,
            0,
            2*cf*lf/iz
        )
    else:
        return np.array([[0, 2*cf/m, 0, 2*cf*lf/iz]]).T
    
    
def get_B_ref_cont(
    vehicle_params: dict[str, float | ca.SX] | None = VehicleParams()
) -> np.ndarray | ca.SX:
    cf = vehicle_params.cf
    cr = vehicle_params.cr
    m = vehicle_params.m
    vx = vehicle_params.vx
    lf = vehicle_params.lf
    lr = vehicle_params.lr
    iz = vehicle_params.iz

    row_2 = -(2*cf*lf-2*cr*lr)/(m*vx)-vx
    row_4 = -(2*cf*lf**2+2*cr*lr**2)/(iz*vx)

    if contains_symbolics(vehicle_params):
        return ca.vertcat(
            0,
            row_2,
            0,
            row_4
        )
    else:
        return np.array([[0, row_2, 0, row_4]]).T
    
    
def get_continuous_system(vehicle_params: dict[str, float | ca.SX] | None = VehicleParams()) -> tuple[np.ndarray | ca.SX, np.ndarray | ca.SX]:
    A_cont = get_A_cont(vehicle_params=vehicle_params)
    B_steer_cont = get_B_steer_cont(vehicle_params=vehicle_params)
    B_ref_cont = get_B_ref_cont(vehicle_params=vehicle_params)
    if contains_symbolics(B_steer_cont) or contains_symbolics(B_ref_cont):
        B_cont = ca.horzcat(B_steer_cont, B_ref_cont)
    else:
        B_cont = np.hstack((B_steer_cont, B_ref_cont))  # reference becomes part of the input (fixed at runtime)
    return A_cont, B_cont


def get_discrete_system(vehicle_params: dict[str, float | ca.SX] | None = VehicleParams(), dt: float = 0.05) -> tuple[np.ndarray | ca.SX, np.ndarray | ca.SX]:
    A_cont, B_cont = get_continuous_system(vehicle_params=vehicle_params)
    if contains_symbolics(A_cont) or contains_symbolics(B_cont):
        sysd = cont2discrete_symbolic(A=A_cont, B=B_cont, dt=dt, method="bilinear")
    else:
        sysd = cont2discrete(system=(A_cont, B_cont, np.eye(4), np.zeros(B_cont.shape)), dt=dt, method="bilinear")
    A_disc = sysd[0]
    B_disc = sysd[1]
    return A_disc, B_disc

def get_bounds(vehicle_params: dict[str, float | ca.SX] | None = VehicleParams()) -> tuple[float | ca.SX]:
    """Get state and action bounds for the specific vehicle."""
    x_lb = np.asarray([[-2], [-10], [-45*np.pi/180], [-100]])
    x_ub = np.asarray([[-lb[0]] for lb in x_lb])

    a_ub = vehicle_params.sw_max / vehicle_params.isw
    a_lb = -a_ub

    e_ub = 0.03 * vehicle_params.vx  # max. curvature for DLC * vehicle speed
    e_lb = -e_ub

    return x_lb, x_ub, a_lb, a_ub, e_lb, e_ub



