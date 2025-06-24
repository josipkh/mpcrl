# %% 
# Imports
# -------
"""
This example applies the DPG-LSTDQ algorithm to the vehicle steering problem.
The LTI system model is based on section 2.5 (eq. 2.45) in "Rajamani - Vehicle Dynamics and Control":
https://link.springer.com/book/10.1007/978-1-4614-1433-9
States are the lateral and heading errors and their derivatives (x = [ey ey_dot epsi epsi_dot]).
The action is the (wheel) steering angle (u = delta).
The reference yaw rate is modelled as a random disturbance (can also be fixed at runtime).
Additionally, the constant road bank angle represents a constant disturbance.
The task can be either:
- a straight drive (continuous, esentially a stabilization problem)
- a double lane change (episodic, following a predefined maneuver in the XY coordinates)
"""

import logging
from typing import Any, Optional

import casadi as cs
import gymnasium as gym
import numpy as np
import numpy.typing as npt
from csnlp import Nlp
from csnlp.wrappers import Mpc
from gymnasium.spaces import Box
from gymnasium.wrappers import TimeLimit

from mpcrl import (
    LearnableParameter,
    LearnableParametersDict,
    LstdDpgAgent,
    UpdateStrategy,
)
from mpcrl import exploration as E
from mpcrl.optim import GradientDescent
from mpcrl.util.control import dlqr
from mpcrl.wrappers.agents import Log, RecordUpdates
from mpcrl.wrappers.envs import MonitorEpisodes

from vehicle_model import (
    vehicle_configs,
    get_discrete_system,
    get_bounds,
    get_cost_matrices,
    get_nondim_matrices,
)

from utils import get_double_lane_change_data, contains_symbolics
from test_configs import experiment_configs

experiment_config = experiment_configs["test"]
dimensionless = experiment_config["dimensionless"]  # set to True to use the dimensionless formulation

vehicle_size = experiment_config["env"]["vehicle_size"]
vehicle_params = vehicle_configs[vehicle_size]
use_learned_parameters = experiment_config["use_learned_parameters"]  # to test the learned (dimensionless) policy transfer
if dimensionless:
    Mx, Mu, Mt = get_nondim_matrices(vehicle_params=vehicle_params)  # x(physical) = Mx * x(dimensionless)
    Mx_inv = np.linalg.inv(Mx)
    Mu_inv = np.linalg.inv(Mu)
    Mt_inv = np.linalg.inv(Mt)

trajectories = []  # for logging the trajectories during learning


# %%
# Defining the environment
# ------------------------

class LtiSystem(gym.Env[npt.NDArray[np.floating], float]):
    """A simple discrete-time LTI system affected by uniform noise."""

    ns = 4  # number of states
    na = 1  # number of actions
    A, B = get_discrete_system(vehicle_params=vehicle_params, method="bilinear")  # dynamics matrices, dimensional
    s_lb, s_ub, a_lb, a_ub, e_lb, e_ub = get_bounds(vehicle_params=vehicle_params)  # bounds of state, action and disturbance
    s_bnd = (s_lb, s_ub)  # bounds of state
    a_bnd = (a_lb, a_ub)  # bounds of control input
    e_bnd = (e_lb, e_ub)  # uniform noise bounds
    Q, R, w = get_cost_matrices(vehicle_params=vehicle_params)  # quadratic cost matrices

    # make the reward dimensionless if needed
    if dimensionless:
        Q = Mx.T @ Q @ Mx
        R = Mu.T @ R @ Mu
        w = Mx @ w

    # define the observation space (primarily for episode termination)
    observation_space = Box(*s_bnd, (ns,1), np.float64)

    # extremely recommended to bound the action space with additive exploration so that
    # we can clip the action before applying it to the system
    action_space = Box(*a_bnd, (na,), np.float64)

    X = 0.0  # global position of the vehicle
    trajectory = []  # current trajectory log

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[npt.NDArray[np.floating], dict[str, Any]]:
        """Resets the state of the LTI system."""
        super().reset(seed=seed, options=options)
        match experiment_config["maneuver"]:
            case "straight":
                ey0 = 1.0 if vehicle_size == "large" else 0.1
            case "double_lane_change":
                ey0 = 0.0
            case _:
                raise ValueError("Unknown maneuver specified.")
        s0 = [ey0, 0.0, 0.0, 0.0]
        self.s = np.asarray(s0).reshape(self.ns, 1)
        self.X = 0.0  # reset the long. position
        self.trajectory = []
        self.trajectory.append(np.vstack((self.s.copy(), 0.0)))  # initial action is zero
        return self.s, {}

    def get_stage_cost(self, state: npt.NDArray[np.floating], action: float) -> float:
        """Computes the stage cost :math:`L(s,a)`."""
        lb, ub = self.s_bnd

        if dimensionless:
            lb, ub = Mx_inv @ lb, Mx_inv @ ub

        return (
            (
                state.T @ self.Q @ state
                + self.R * action**2
                + self.w.T @ np.maximum(0, lb - state)
                + self.w.T @ np.maximum(0, state - ub)
            ).item()
        )

    def step(
        self, action: cs.DM
    ) -> tuple[npt.NDArray[np.floating], float, bool, bool, dict[str, Any]]:
        """Steps the LTI system."""
        action = np.asarray(action)
        if not self.action_space.contains(action.ravel()):
            print(f"WARNING: Action {action.item():.4f} is out of bounds ({self.action_space.low[0]:.4f}, {self.action_space.high[0]:.4f})")
        action = action.item()

        if dimensionless:
            action = (Mu * action).item()  # transform to dimensional action

        yaw_rate_ref = self.get_yaw_rate_ref(horizon=3)[0]  # get the yaw rate reference (disturbance)
        if yaw_rate_ref.size > 1:
            yaw_rate_ref = yaw_rate_ref[0]  # get only the first value
        s_new = self.A @ self.s + self.B @ np.asarray([[action], [yaw_rate_ref]])

        # add road bank effect (constant disturbance; 5 deg for "large" -> 0.106 dimensionless)
        road_bank_angle = experiment_config["env"]["road_bank_angle"]  # [deg]
        road_bank_angle = np.deg2rad(road_bank_angle)
        g = 9.81  # gravity
        s_new += np.asarray([[0.0], [g], [0.0], [0.0]]) * np.sin(road_bank_angle)
        self.trajectory.append(np.vstack((s_new,action)))

        # update the global longitudinal position
        self.X = self.X + vehicle_params["vx"] * vehicle_params["dt"]  # acceptably wrong (assumes small steering angles)

        # check if the new state is within bounds
        state_out_of_bounds = not self.observation_space.contains(s_new)
        end_of_maneuver = experiment_config["maneuver"] == "double_lane_change" and self.X >= 200  # [m] maneuver limit
        terminated = state_out_of_bounds or end_of_maneuver
        if terminated:
            trajectories.append(self.trajectory)  # log the trajectory before termination
            if state_out_of_bounds:
                out_of_bounds = np.logical_or(s_new < self.s_lb, s_new > self.s_ub)
                state_labels = ["ey", "ey_dot", "epsi", "epsi_dot"]
                for k, is_out in enumerate(out_of_bounds):
                    if is_out:
                        print(f"WARNING: State {state_labels[k]} is out of bounds, terminating episode...")
            elif end_of_maneuver:
                print("INFO: End of maneuver reached, terminating episode...")
        truncated = False
        info = {}

        self.s = s_new  # keep this physical
        if dimensionless:
            s_new = Mx_inv @ s_new  # transform to dimensionless state
            action = (Mu_inv * action).item()  # transform back to dimensionless action

        r = self.get_stage_cost(s_new, action)  # use the dimensionless state for the reward

        return s_new, r, terminated, truncated, info
    

    def get_yaw_rate_ref(self, horizon: int) -> npt.NDArray[np.floating]:
        """Returns the yaw rate reference (preview) along the horizon."""
        match experiment_config["maneuver"]:
            case "straight":
                road_curvature = np.zeros((1, horizon))
            case "double_lane_change":
                road_curvature = get_double_lane_change_data(self.X, horizon, vehicle_params)[0]
            case _:
                raise ValueError("Unknown maneuver specified.")
        
        return vehicle_params["vx"] * road_curvature  # eq. (2.38) in Rajamani
    

# %%
# Defining the MPC controller
# ---------------------------

# MPC formulation similar to Sec. VII.A in https://ieeexplore.ieee.org/document/8701462
class LinearMpc(Mpc[cs.SX]):
    """A simple linear MPC controller."""

    horizon = 10
    discount_factor = 0.99
    nx, nu = LtiSystem.ns, LtiSystem.na  # number of states and actions

    # fixed system matrices (initial guess, not learnable)
    if dimensionless:
        A_fixed, B_fixed = get_discrete_system(vehicle_params=vehicle_params, method="dimensionless")
    else:
        A_fixed, B_fixed = get_discrete_system(vehicle_params=vehicle_params, method="bilinear")

    if use_learned_parameters:
        examples_folder = '/home/josip/mpcrl/examples/jkh/vehicle_steering'
        output_folder = 'output_2025-06-10_18-37-56-large-learned'  # 'output_2025-06-05_11-28-35-small-learned'
        file_name = 'learned_parameters.npz'
        learned_parameters = np.load(examples_folder + '/' + output_folder + '/' + file_name)
        learnable_pars_init = {key: value for key, value in learned_parameters.items()}
    else:
        learnable_pars_init = {
            # "V0": np.zeros(nx),  # cost modification, V0*x0
            # "x_lb": np.zeros(nx),  # constraint backoff
            # "x_ub": np.zeros(nx),  # constraint backoff
            "b": np.zeros(nx),  # affine term in the dynamics
            "f": np.zeros(nx + nu),  # affine term in the cost
            # "A": A_fixed,
            # "B": B_fixed[:,0,np.newaxis],  # just the steering input
            # "cf": 0.5 * np.asarray(vehicle_params["cf"])
        }

    fixed_parameters = {
        "w": np.zeros((1,horizon)),  # fixed disturbance (yaw rate reference)
    }

    def __init__(self) -> None:
        N = self.horizon
        gamma = self.discount_factor
        nx, nu = LtiSystem.ns, LtiSystem.na
        x_bnd, u_bnd = LtiSystem.s_bnd, LtiSystem.a_bnd
        nlp = Nlp[cs.SX]()
        super().__init__(nlp, N)

        # parameters
        # V0 = self.parameter("V0", (nx,))
        # x_lb = self.parameter("x_lb", (nx,))
        # x_ub = self.parameter("x_ub", (nx,))
        x_lb, x_ub = np.zeros_like(x_bnd[0]), np.zeros_like(x_bnd[0])
        b = self.parameter("b", (nx, 1))
        f = self.parameter("f", (nx + nu, 1))
        # A = self.parameter("A", (nx, nx))
        # B = self.parameter("B", (nx, nu))
        cf = self.parameter("cf", (1, 1))

        # modify the parameter dictionary with learnable ones
        # TODO: handle a learnable cost
        mpc_vehicle_params = vehicle_params.copy()  # a copy which might contain symbolics
        for key in self.learnable_pars_init.keys():
            if key in mpc_vehicle_params.keys():
                mpc_vehicle_params[key] = locals()[key]  # assign the related parameter (same name must be used)
                # TODO * vehicle_params[key], [0-1] * true_value ?

        # (possibly) learnable system matrices
        if dimensionless:
            A_learnable, B_learnable = get_discrete_system(vehicle_params=mpc_vehicle_params, method="dimensionless")
        else:
            A_learnable, B_learnable = get_discrete_system(vehicle_params=mpc_vehicle_params, method="bilinear")

        # variables (state, action, slack)
        x, _ = self.state("x", nx, bound_initial=False)
        u, _ = self.action("u", nu, lb=u_bnd[0], ub=u_bnd[1])
        s, _, _ = self.variable("s", (nx, N), lb=0)
        _ = self.disturbance("w", 1)  # for the yaw rate reference

        # dynamics (x_+ = A x + B u + D w + c)
        B = B_learnable[:,0] if contains_symbolics(B_learnable) else B_learnable[:,0,np.newaxis]  # steering input
        D = B_learnable[:,1] if contains_symbolics(B_learnable) else B_learnable[:,1,np.newaxis]  # yaw rate reference
        self.set_affine_dynamics(A=A_learnable, B=B, D=D, c=b)

        # other constraints
        self.constraint("x_lb", x_bnd[0] + x_lb - s, "<=", x[:, 1:])
        self.constraint("x_ub", x[:, 1:], "<=", x_bnd[1] + x_ub + s)

        du = u[:, 1:] - u[:, 0:-1]
        du_ub = LtiSystem.a_bnd[1] * vehicle_params["dt"]  # rate limit, 0 to max in 1 second
        self.constraint("du_lb", du, ">=", -du_ub)
        self.constraint("du_ub", du, "<=",  du_ub)

        # objective
        Q, R, w = get_cost_matrices(vehicle_params=vehicle_params)
        if dimensionless:
            Q = Mx.T @ Q @ Mx
            R = Mu.T @ R @ Mu
            w = Mx @ w
        S = cs.DM(dlqr(self.A_fixed, self.B_fixed[:,0,np.newaxis], Q, R)[1])  # terminal cost matrix, calculated with the initial guess for A, B
        gammapowers = cs.DM(gamma ** np.arange(N)).T
        objective = 0.0
        for k in range(N):
            objective += gammapowers[k] * (cs.bilin(Q, x[:, k]) + R * u[k]**2 + w.T @ s[:, k])  # quadratic stage cost
            objective += gammapowers[k] * (f.T @ cs.vertcat(x[:, k], u[k]))  # linear stage cost
        objective += gamma**N * cs.bilin(S, x[:, -1])  # terminal cost
        # TODO: check about V0
        # V0 = cs.DM(np.zeros(nx))
        # objective += V0.T @ x[:, 0]  # cost modification; to have a derivative, V0 must be a function of the state
        self.minimize(objective)

        # solver
        solver = "osqp"  # choose one of the solvers shipped with CasADi
        match solver:
            case "ipopt":
                opts = {
                "expand": True,
                "print_time": False,
                "bound_consistency": True,
                "calc_lam_p": False,
                "ipopt": {"max_iter": 500, "print_level": 0},
                }
            case "fatrop":
                opts = {
                "expand": True,
                "print_time": False,
                "bound_consistency": True,
                "calc_lam_p": False,
                "fatrop": {"max_iter": 500, "print_level": 0},
                }
                # additional options from the fatrop demo (TODO: modify csnlp)
                # opts["structure_detection"] = "auto",
                # opts["debug"] = True
                # opts["equality"] = [True for _ in range(N * x.numel())]  # TODO: add False for inequalities

                # codegen of helper functions
                # opts["jit"] = True
                # opts["jit_temp_suffix"] = False
                # opts["jit_options"] = {"flags": ["-O3"],"compiler": "ccache gcc"}
            case "osqp":
                opts = {"osqp": {"verbose": False}}  # , "error_on_fail": False
            case "qpoases":
                opts = {}
                # opts = {"qpoases": {"printLevel": "none"}}  # , "error_on_fail": False  # TODO: figure out why this doesn't work

        self.init_solver(opts, solver=solver, type="nlp" if solver in ["ipopt", "fatrop"] else "conic")


# %% Defining the agent (subclass of LstdDpgAgent, with updating of the fixed parameters)
# ---------------------------------------------------------------------------------------

class MyLstdDpgAgent(LstdDpgAgent[cs.SX, float]):
    def __init__(self, mpc: Mpc[cs.SX], *args: Any, **kwargs: Any) -> None:
        super().__init__(mpc, *args, **kwargs)
        self._horizon = mpc.prediction_horizon

    def on_episode_start(self, env: LtiSystem, episode: int, state: np.ndarray) -> None:
        super().on_episode_start(env, episode, state)
        self.update_yaw_rate_ref(env)

    def on_timestep_end(self, env: LtiSystem, episode: int, timestep: int) -> None:
        super().on_timestep_end(env, episode, timestep)
        self.update_yaw_rate_ref(env)

    def update_yaw_rate_ref(self, env: LtiSystem) -> None:
        yaw_rate_ref = env.unwrapped.get_yaw_rate_ref(self._horizon)
        if dimensionless:
            yaw_rate_ref = Mt * yaw_rate_ref
        self.fixed_parameters.update(zip("w", yaw_rate_ref))


# %%
# Simulation
# ----------

if __name__ == "__main__":
    # instantiate the env and wrap it
    env = MonitorEpisodes(TimeLimit(LtiSystem(), max_episode_steps=experiment_config["max_episode_steps"]))

    # now build the MPC and the dict of learnable parameters
    mpc = LinearMpc()
    learnable_parameters = LearnableParametersDict[cs.SX](
        (
            LearnableParameter(name, val.shape, val, sym=mpc.parameters[name])
            for name, val in mpc.learnable_pars_init.items()
        )
    )

    # build and wrap appropriately the agent
    match experiment_config["maneuver"]:
        case "double_lane_change":
            rollout_length = -1  # will take the whole episode
            update_strategy = UpdateStrategy(1, "on_episode_end")  # episodic
        case "straight":
            rollout_length = 100  # we have to update at some point
            update_strategy = UpdateStrategy(rollout_length, "on_timestep_end")  # continuous
    agent = Log(
        RecordUpdates(
            MyLstdDpgAgent(
                mpc=mpc,
                learnable_parameters=learnable_parameters,
                fixed_parameters=mpc.fixed_parameters,
                discount_factor=mpc.discount_factor,
                optimizer=GradientDescent(learning_rate=experiment_config["learning_rate"]),
                update_strategy=update_strategy,
                rollout_length=rollout_length,
                exploration=E.OrnsteinUhlenbeckExploration(0.0, 0.05*LtiSystem.a_bnd[1], mode="additive"),
                record_policy_performance=True,
                record_policy_gradient=True,
            )
        ),
        level=logging.DEBUG,
        log_frequencies={"on_timestep_end": 1000},
    )

    # launch the training simulation
    agent.train(env=env, episodes=experiment_config["episodes"], seed=0)


    # %%
    # Display the results
    # -------------------
    import matplotlib.pyplot as plt
    from plotting import (
        plot_trajectory_error_frame,
        plot_performance,
        plot_parameters,
        plot_trajectories
    )

    X = env.get_wrapper_attr("observations")[0].squeeze().T
    U = env.get_wrapper_attr("actions")[0].squeeze()
    R = env.get_wrapper_attr("rewards")[0]

    # scale the logs back to the physical values if needed
    # the reward is equivalent in both cases (if originally dimensionless)
    if dimensionless:
        X = Mx @ X
        U = (Mu * U).ravel()

    match experiment_config["maneuver"]:
        case "straight":
            fig1 = plot_trajectory_error_frame(X, U, vehicle_params)
        case "double_lane_change":
            fig1 = plot_trajectories(trajectories, vehicle_params)
    fig2 = plot_performance(agent, R)
    fig3 = plot_parameters(agent)

    plt.show(block=False)

    user_input = input("Do you want to save the output? (y/[n]): ").strip().lower()

    if user_input == "y":
        import datetime
        import os

        # create a timestamped folder
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        main_folder_path = "/home/josip/mpcrl/examples/jkh/vehicle_steering"
        new_folder_path = f"output_{timestamp}"
        folder_path = os.path.join(main_folder_path, new_folder_path)
        folder_path += f"-{vehicle_size}"  # append vehicle size
        folder_path += "-transfer" if use_learned_parameters else "-learned"
        os.makedirs(folder_path, exist_ok=True)

        # save the figures
        fig1.savefig(os.path.join(folder_path, "trajectory.pdf"))
        fig2.savefig(os.path.join(folder_path, "performance.pdf"))
        fig3.savefig(os.path.join(folder_path, "parameters.pdf"))

        # save the final (learned) parameters
        np.savez(
            os.path.join(folder_path, "learned_parameters.npz"),
            **{name: val[-1] for name, val in agent.updates_history.items()}
        )

        print(f"Output saved in folder: {folder_path}")
    else:
        print("Output not saved.")
    plt.close()
