import numpy as np
import casadi as ca
from dataclasses import fields, is_dataclass


def contains_symbolics(object) -> bool:
    """
    Checks if an object contains CasADi symbolics
    """
    casadi_types = (ca.SX, ca.MX, ca.DM)
    if is_dataclass(object):
        for field in fields(object):
            field_value = getattr(object, field.name)
            if isinstance(field_value, casadi_types):
                return True
        return False
    elif isinstance(object, np.ndarray):
        return False
    elif isinstance(object, ca.SX) or isinstance(object, ca.MX) or isinstance(object, ca.DM):
        return True
    elif isinstance(object, dict):
        for _, value in object.items():
            if isinstance(value, casadi_types):
                return True
        return False
    else:
        raise TypeError("unknown object type")
    

def cont2discrete_symbolic(A: ca.SX, B: ca.SX, dt: float, method: str) -> tuple[ca.SX, ca.SX]:
    """
    Discretizes a continuous-time LTI system, described by matrices A and B, using the selected method.    
    """

    if method == "bilinear":
        # Identity matrix of appropriate dimensions
        I = ca.SX.eye(A.shape[0])

        # Compute the discretized A and B using the Tustin (Bilinear) method
        A_d = ca.mtimes(ca.inv(I + (A * dt / 2)), I - (A * dt / 2))
        B_d = ca.mtimes(ca.inv(I + (A * dt / 2)), B * dt)
    elif method == "euler":
        I = ca.SX.eye(A.shape[0])
        A_d = I + A * dt
        B_d = B * dt
    else:
        raise NotImplementedError(f"Discretization method {method} not implemented")
    return A_d, B_d


def compute_curvature(p_xy: np.ndarray, dt: float) -> np.ndarray:
    """
    :param p_xy: array of size (n,2) representing Cartesian 2D points
    :return: curvature and path length
    """
    assert p_xy.shape[1] == 2, "Input array must have shape (n,2)"
    assert p_xy.shape[0] > 2, "Input array must have at least 3 points"

    # first derivatives
    dx = np.diff(p_xy[:, 0]) / dt
    dy = np.diff(p_xy[:, 1]) / dt
    dx = np.hstack((dx, dx[-1]))  # extend last value to match original length
    dy = np.hstack((dy, dy[-1]))

    # second derivatives
    d2x = np.diff(dx) / dt
    d2y = np.diff(dy) / dt
    d2x = np.hstack((d2x, d2x[-1]))
    d2y = np.hstack((d2y, d2y[-1]))

    # calculation of curvature from the typical formula (Wikipedia)
    curvature = (dx * d2y - d2x * dy) / (dx * dx + dy * dy) ** 1.5
    # path_length = np.cumsum(np.sqrt(dx ** 2 + dy ** 2))

    return curvature


def get_double_lane_change_data(X: np.ndarray, horizon: int, vehicle_params: dict[str, float | ca.SX]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates road curvature, lateral position and yaw angle for a double lane change maneuver."""

    X += vehicle_params["vx"] * vehicle_params["dt"] * np.asarray(list(range(horizon)))  # extend X along the horizon

    large_car_length = 2.622 # [m], from a reference large vehicle
    current_car_length = (vehicle_params["lf"] + vehicle_params["lr"])
    normalization_factor = current_car_length / large_car_length

    shape = 2.4
    dx1 = 25 * normalization_factor
    dx2 = 21.95 * normalization_factor
    dy1 = 4.05 * normalization_factor
    dy2 = 5.7 * normalization_factor
    Xs1 = 27.19 * normalization_factor
    Xs2 = 56.46 * normalization_factor

    z1 = shape/dx1*(X - Xs1) - shape/2
    z2 = shape/dx2*(X - Xs2) - shape/2

    # from eq. (20) in https://ieeexplore.ieee.org/document/10308482
    # originally from eq. (16) in https://sci-hub.st/10.1504/ijvas.2005.008237
    # NOTE: expressions for Y and psi are switched in the papers
    Y = dy1/2*(1+np.tanh(z1)) - dy2/2*(1+np.tanh(z2))  # (min, max) = (-1.65, 3.52)
    psi = np.arctan(dy1 * (1 / np.cosh(z1))**2 * (1.2 / dx1) - dy2 * (1 / np.cosh(z2))**2 * (1.2 / dx2))

    XY = np.vstack((X,Y)).T
    road_curvature = compute_curvature(p_xy=XY, dt=vehicle_params["dt"])  # (min, max) = (-0.024257, 0.027217)

    return road_curvature, Y, psi


def frenet2inertial(
    e1: np.ndarray, e2: np.ndarray, yaw_ref: np.ndarray, vx: float, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts a trajectory from the body-fixed to the inertial coordinate frame.
    Based on section 2.7 in https://link.springer.com/book/10.1007/978-1-4614-1433-9
    Constant longitudinal speed is assumed.
    """
    x_int = vx * dt * np.cumsum(np.cos(yaw_ref))
    x = x_int - e1 * np.sin(e2 + yaw_ref)

    y_int = vx * dt * np.cumsum(np.sin(yaw_ref))
    y = y_int + e1 * np.cos(e2 + yaw_ref)

    return x, y
