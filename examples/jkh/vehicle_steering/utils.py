import numpy as np
import casadi as ca
from dataclasses import fields, is_dataclass

# TODO: typehint for dataclass
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
    elif isinstance(object, ca.SX):
        return True
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
