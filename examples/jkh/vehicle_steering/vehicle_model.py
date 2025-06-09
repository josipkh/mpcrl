"""
This module contains the dynamics model for the vehicle steering problem.
The system model is based on section 2.5 (eq. 2.45) in "Rajamani - Vehicle Dynamics and Control".
States are the lateral and heading errors and their derivatives (x = [ey ey_dot epsi epsi_dot]).
The action is the (wheel) steering angle (u = delta).
The reference yaw rate is modelled as an additional control input (will be fixed at runtime).
"""

import numpy as np
import casadi as ca
from scipy.signal import cont2discrete
from dataclasses import dataclass, field
from utils import contains_symbolics, cont2discrete_symbolic


vehicle_configs = {
    "large": {
        "cf": 63271.7,  # [N/rad]   front cornering stiffness (for one tire)
        "cr": 63271.7,  # [N/rad]   rear cornering stiffness (for one tire)
        "m": 1600.0,    # [kg]      vehicle mass
        "vx": 15.6,     # [m/s]     longitudinal vehicle speed
        "lf": 1.1578,   # [m]       distance from the center of gravity to the front axle
        "lr": 1.4642,   # [m]       distance from the center of gravity to the rear axle
        "iz": 2675.7,   # [kg*m^2]  vehicle moment of inertia
        "isw": 13,      # [-]       steering ratio
        "sw_max": float(np.deg2rad(90))  # [rad] maximum steering wheel angle (wheel angle = steering wheel angle / steering ratio)
    },
    "small": {
        "cf": 8.25,     # [N/rad]   front cornering stiffness (for one tire)
        "cr": 8.25,     # [N/rad]   rear cornering stiffness (for one tire)
        "m": 2.173,     # [kg]      vehicle mass
        "vx": 1.5,      # [m/s]     longitudinal vehicle speed
        "lf": 0.1115,   # [m]       distance from the center of gravity to the front axle
        "lr": 0.141,    # [m]       distance from the center of gravity to the rear axle
        "iz": 0.0337,   # [kg*m^2]  vehicle moment of inertia
        "isw": 13,      # [-]       steering ratio
        "sw_max": float(np.deg2rad(90))  # [rad] maximum steering wheel angle (wheel angle = steering wheel angle / steering ratio)
    }
}


@dataclass(kw_only=True)
class VehicleParams:
    vehicle_size: str
    cf:     float = field(default=None)
    cr:     float = field(default=None)
    m:      float = field(default=None)
    vx:     float = field(default=None)
    lf:     float = field(default=None)
    lr:     float = field(default=None)
    iz:     float = field(default=None)
    isw:    float = field(default=None)
    sw_max: float = field(default=None)

    def __post_init__(self):
        if self.vehicle_size is not None:
            config = vehicle_configs[self.vehicle_size]
            self.cf = config['cf']
            self.cr = config['cr']
            self.m = config['m']
            self.vx = config['vx']
            self.lf = config['lf']
            self.lr = config['lr']
            self.iz = config['iz']
            self.isw = config['isw']
            self.sw_max = config['sw_max']
        else:
            raise ValueError("vehicle_size must be specified in VehicleParams. Use 'large' or 'small'.")


def get_A_cont(
    vehicle_params: dict[str, float | ca.SX] | None
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
    vehicle_params: dict[str, float | ca.SX] | None
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
    vehicle_params: dict[str, float | ca.SX] | None
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
    
    
def get_continuous_system(vehicle_size: str) -> tuple[np.ndarray | ca.SX, np.ndarray | ca.SX]:
    vehicle_params = VehicleParams(vehicle_size=vehicle_size)
    A_cont = get_A_cont(vehicle_params=vehicle_params)
    B_steer_cont = get_B_steer_cont(vehicle_params=vehicle_params)
    B_ref_cont = get_B_ref_cont(vehicle_params=vehicle_params)
    if contains_symbolics(B_steer_cont) or contains_symbolics(B_ref_cont):
        B_cont = ca.horzcat(B_steer_cont, B_ref_cont)
    else:
        B_cont = np.hstack((B_steer_cont, B_ref_cont))  # reference becomes part of the input (fixed at runtime)
    return A_cont, B_cont


def get_discrete_system(vehicle_size: str, dt: float = 0.05, method: str = "bilinear") -> tuple[np.ndarray | ca.SX, np.ndarray | ca.SX]:
    A_cont, B_cont = get_continuous_system(vehicle_size=vehicle_size)
    if method == "dimensionless":
        Mx, Mu, Mt = get_nondim_matrices(vehicle_size=vehicle_size)
        Mx_inv = np.linalg.inv(Mx)
        A_cont = Mt * Mx_inv @ A_cont @ Mx
        B_cont = Mt * Mx_inv @ B_cont * Mu
        dt = (np.linalg.inv(Mt) * dt).item()
        sysd = cont2discrete(system=(A_cont, B_cont, np.eye(4), np.zeros(B_cont.shape)), dt=dt, method="bilinear")
    elif method == "bilinear":
        if contains_symbolics(A_cont) or contains_symbolics(B_cont):
            sysd = cont2discrete_symbolic(A=A_cont, B=B_cont, dt=dt, method="bilinear")
        else:
            sysd = cont2discrete(system=(A_cont, B_cont, np.eye(4), np.zeros(B_cont.shape)), dt=dt, method="bilinear")
    else:
        raise ValueError(f"Method {method} not supported. Use 'bilinear' or 'dimensionless'.")
    A_disc = sysd[0]
    B_disc = sysd[1]
    return A_disc, B_disc


def get_bounds(vehicle_size: str) -> tuple[float | ca.SX]:
    """Get state and action bounds for the specific vehicle."""
    vehicle_params = VehicleParams(vehicle_size=vehicle_size)

    ey_ub = (vehicle_params.lf + vehicle_params.lr) / 2
    dey_ub = 0.5 * vehicle_params.vx
    epsi_ub = np.deg2rad(45)
    depsi_ub = 100 * vehicle_params.vx / (vehicle_params.lf + vehicle_params.lr)  # pretty large

    s_ub = np.asarray([[ey_ub], [dey_ub], [epsi_ub], [depsi_ub]])
    s_lb = np.asarray([[-ub[0]] for ub in s_ub])

    a_ub = vehicle_params.sw_max / vehicle_params.isw
    a_lb = -a_ub

    e_ub = 0.03 * vehicle_params.vx  # max. curvature for DLC * vehicle speed
    e_lb = -e_ub

    return s_lb, s_ub, a_lb, a_ub, e_lb, e_ub


def get_cost_matrices(vehicle_size: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns the quadratic cost matrices for the controller (x'Qx + u'Ru)."""
    if vehicle_size == "large":
        Q = np.diag([1, 1e-3, 1, 1e-3])
        R = np.diag([1])
        w = np.asarray([[1e2], [1e2], [1e2], [1e2]])  # penalty weight for bound violations
    elif vehicle_size == "small":
        Q = np.diag([107.83, 0.10816, 1, 1e-3])  # q = (Mx @ mx_inv).T @ Q @ (Mx @ mx_inv)
        R = np.diag([1])
        w = np.asarray([[1038.416], [1040], [1e2], [1e2]])  # w.T = W.T @ (Mx @ mx_inv)
    else:
        raise ValueError(f"MPC weights for size {vehicle_size} not specified. Use 'large' or 'small'.")
    return Q, R, w


def get_nondim_matrices(vehicle_size: str) -> tuple[np.ndarray]:
    """Returns the matrices for transforming the system to a non-dimensional form."""
    vehicle_params = VehicleParams(vehicle_size=vehicle_size)
    vx = vehicle_params.vx
    L = vehicle_params.lf + vehicle_params.lr

    Mx = np.diag([L, vx, 1.0, vx/L])  # state is [ey, ey_dot, epsi, epsi_dot]
    Mu = np.diag([1.0])  # input is an angle, no transformation needed
    Mt = np.diag([L/vx])  # time transformation
    return Mx, Mu, Mt 


if __name__ == "__main__":
    # Example usage
    vehicle_size = "large"
    dt = 0.05
    A_disc, B_disc = get_discrete_system(vehicle_size=vehicle_size, dt=dt)
    print("Discrete A matrix:\n", A_disc)
    print("Discrete B matrix:\n", B_disc)

    s_lb, s_ub, a_lb, a_ub, e_lb, e_ub = get_bounds(vehicle_size=vehicle_size)
    print("State bounds:", s_lb, s_ub)
    print("Action bounds:", a_lb, a_ub)
    print("Error bounds:", e_lb, e_ub)

    Q, R, w = get_cost_matrices(vehicle_size=vehicle_size)
    print("Cost matrices Q:\n", Q)
    print("Cost matrices R:\n", R)
    print("Penalty weights w:\n", w)
