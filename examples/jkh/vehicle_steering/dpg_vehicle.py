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
The resulting task is esentially a stabilization problem (LQR).
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
    VehicleParams,
    get_discrete_system,
    get_bounds,
    get_cost_matrices,
    get_nondim_matrices,
)

from experiment_configs import configs

experiment_config = configs["large_transfer"]  # or "small_learn", "large_learn", "test"

dimensionless = True  # set to True to use the dimensionless approach
vehicle_size = experiment_config["vehicle_size"]
use_learned_parameters = experiment_config["use_learned_parameters"]  # to test the learned (dimensionless) policy transfer
if dimensionless:
    Mx, Mu, Mt = get_nondim_matrices(vehicle_size=vehicle_size)  # x(physical) = Mx * x(dimensionless)
    Mx_inv = np.linalg.inv(Mx)
    Mu_inv = np.linalg.inv(Mu)
    Mt_inv = np.linalg.inv(Mt)

# %%
# Defining the environment
# ------------------------
# First things first, we need to build the environment. We will use the :mod:`gymnasium`
# library to do so. The most important methods are :func:`gymnasium.Env.reset` and
# :func:`gymnasium.Env.step`, which will be called to reset the environment to its
# initial state and to step the dynamics and receive a realization of the reward signal,
# respectively. The environment is defined as a the following class.


class LtiSystem(gym.Env[npt.NDArray[np.floating], float]):
    """A simple discrete-time LTI system affected by uniform noise."""

    ns = 4  # number of states
    na = 1  # number of actions
    A, B = get_discrete_system(vehicle_size=vehicle_size)  # dynamics matrices
    s_lb, s_ub, a_lb, a_ub, e_lb, e_ub = get_bounds(vehicle_size=vehicle_size)  # bounds of state, action and disturbance
    s_bnd = (s_lb, s_ub)  # bounds of state
    a_bnd = (a_lb, a_ub)  # bounds of control input
    e_bnd = (e_lb, e_ub)  # uniform noise bounds
    Q, R, w = get_cost_matrices(vehicle_size=vehicle_size)  # quadratic cost matrices

    # make the reward dimensionless if needed
    if dimensionless:
        Q = Mx.T @ Q @ Mx
        R = Mu.T @ R @ Mu
        w = Mx @ w

    # extremely recommended to bound the action space with additive exploration so that
    # we can clip the action before applying it to the system
    action_space = Box(*a_bnd, (na,), np.float64)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[npt.NDArray[np.floating], dict[str, Any]]:
        """Resets the state of the LTI system."""
        super().reset(seed=seed, options=options)
        self.s = np.asarray([0.0, 0.0, 0.0, 0.0]).reshape(self.ns, 1)
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
        action = np.asarray(action).item()

        if dimensionless:
            action = (Mu * action).item()  # transform to dimensional action

        disturbance = 0 * self.np_random.uniform(*self.e_bnd)  # road curvature
        s_new = self.A @ self.s + self.B @ np.asarray([[action], [disturbance]])

        # add road bank effect (constant disturbance; 5 deg = 0.106 dimensionless)
        road_bank_angle = np.deg2rad(5)  # up to 5 degrees should be realistic
        g = 9.81 if vehicle_size == "large" else 0.9418  # TODO: remove the hack with gravity scaling
        s_new += np.asarray([[0.0], [g], [0.0], [0.0]]) * np.sin(road_bank_angle)

        self.s = s_new  # keep this physical
        if dimensionless:
            s_new = Mx_inv @ s_new  # transform to dimensionless state
            action = (Mu_inv * action).item()  # transform back to dimensionless action

        r = self.get_stage_cost(s_new, action)  # use the dimensionless state for the reward

        return s_new, r, False, False, {}


# %%
# Defining the MPC controller
# ---------------------------
# The second component is the MPC controller. We'll create a custom that, of course,
# inherits from :class:`csnlp.wrappers.Mpc`. The implementation is as follows, and it is
# in line with the theory presented above.

# MPC formulation similar to Sec. VII.A in https://ieeexplore.ieee.org/document/8701462
class LinearMpc(Mpc[cs.SX]):
    """A simple linear MPC controller."""

    horizon = 10
    discount_factor = 0.9
    dt = 0.05  # [s] sampling time
    nx, nu = LtiSystem.ns, LtiSystem.na  # number of states and actions

    if dimensionless:
        A_init, B_init = get_discrete_system(vehicle_size=vehicle_size, dt=dt, method="dimensionless")
    else:
        A_init, B_init = get_discrete_system(vehicle_size=vehicle_size, dt=dt, method="bilinear")

    if use_learned_parameters:
        examples_folder = '/home/josip/mpcrl/examples/jkh/vehicle_steering'
        output_folder = 'output_2025-05-30_09-59-39-small-learned'
        file_name = 'learned_parameters.npz'
        learned_parameters = np.load(examples_folder + '/' + output_folder + '/' + file_name)
        learnable_pars_init = {key: value for key, value in learned_parameters.items()}
    else:
        learnable_pars_init = {
            "V0": np.zeros(nx),  # cost modification, V0*x0
            "x_lb": np.zeros(nx),  # constraint backoff
            "x_ub": np.zeros(nx),  # constraint backoff
            "b": np.zeros(nx),  # affine term in the dynamics
            "f": np.zeros(nx + nu),  # affine term in the cost
            "A": A_init,
            "B": B_init[:,0,np.newaxis],  # just the steering input
            # "k": np.asarray([1.0]),  # test learning of individual parameters
        }

    def __init__(self) -> None:
        N = self.horizon
        gamma = self.discount_factor
        nx, nu = LtiSystem.ns, LtiSystem.na
        x_bnd, u_bnd = LtiSystem.s_bnd, LtiSystem.a_bnd
        nlp = Nlp[cs.SX]()
        super().__init__(nlp, N)

        # parameters
        V0 = self.parameter("V0", (nx,))
        x_lb = self.parameter("x_lb", (nx,))
        x_ub = self.parameter("x_ub", (nx,))
        b = self.parameter("b", (nx, 1))
        f = self.parameter("f", (nx + nu, 1))
        A = self.parameter("A", (nx, nx))
        B = self.parameter("B", (nx, nu))
        # k = self.parameter("k", (1,1))
        # B = k * LtiSystem.B[:, 0, np.newaxis]  # just the steering input

        # variables (state, action, slack)
        x, _ = self.state("x", nx, bound_initial=False)
        u, _ = self.action("u", nu, lb=u_bnd[0], ub=u_bnd[1])
        s, _, _ = self.variable("s", (nx, N), lb=0)

        # dynamics
        self.set_affine_dynamics(A, B, c=b)

        # other constraints
        self.constraint("x_lb", x_bnd[0] + x_lb - s, "<=", x[:, 1:])
        self.constraint("x_ub", x[:, 1:], "<=", x_bnd[1] + x_ub + s)

        # du = u[:, 1:] - u[:, 0:-1]
        # du_ub = LtiSystem.u_bnd[1] * self.dt  # rate limit, 0 to max in 1 second
        # self.constraint("du_lb", du, ">=", -du_ub)
        # self.constraint("du_ub", du, "<=",  du_ub)

        # objective
        Q, R, w = get_cost_matrices(vehicle_size=vehicle_size)
        if dimensionless:
            Q = Mx.T @ Q @ Mx
            R = Mu.T @ R @ Mu
            w = Mx @ w
        A_init, B_init = self.learnable_pars_init["A"], self.learnable_pars_init["B"]  # LtiSystem.B[:, 0, np.newaxis]
        S = cs.DM(dlqr(A_init, B_init, Q, R)[1])  # terminal cost matrix
        gammapowers = cs.DM(gamma ** np.arange(N)).T
        objective = 0.0
        for k in range(N):
            objective += gammapowers[k] * (cs.bilin(Q, x[:, k]) + R * u[k]**2 + w.T @ s[:, k])  # quadratic stage cost
            objective += gammapowers[k] * (f.T @ cs.vertcat(x[:, k], u[k]))  # linear stage cost
        objective += gamma**N * cs.bilin(S, x[:, -1])  # terminal cost
        objective += V0.T @ x[:, 0]  # cost modification; to have a derivative, V0 must be a function of the state
        self.minimize(objective)

        # solver
        solver = "QP"  # "NLP" or "QP"
        if solver == "NLP":
            opts = {
                "expand": True,
                "print_time": False,
                "bound_consistency": True,
                "calc_lam_p": False,
                "fatrop": {"max_iter": 500, "print_level": 0},
            }
            # additional options from the fatrop demo
            # opts["structure_detection"] = "auto",
            # opts["debug"] = True
            # opts["equality"] = [True for _ in range(N * x.numel())]  # TODO: add False for inequalities

            # codegen of helper functions
            # opts["jit"] = True
            # opts["jit_temp_suffix"] = False
            # opts["jit_options"] = {"flags": ["-O3"],"compiler": "ccache gcc"}

            self.init_solver(opts, solver="fatrop", type="nlp")
        elif solver == "QP":
            opts = {"osqp": {"verbose": False}, "error_on_fail": False}
            self.init_solver(opts, solver="osqp", type="conic")


# %%
# Simulation
# ----------
# So far, we have only defined the classes for the environment and the MPC controller.
# Now, it is time to instantiate these and run the simulation. This is comprised of
# multiple steps, which are detailed below.
#
# 1. We instantiate the environment. Note how it is wrapped in two different wrappers:
#    :class:`gymnasium.wrappers.TimeLimit` is used to impose a maximum amount of steps
#    to be simulated, whereas :class:`mpcrl.wrappers.envs.MonitorEpisodes` is used to
#    record the state, action and reward signals at each time step for plotting
#    purposes.
# 2. We instantiate the MPC controller and define its learnable parameters.
# 3. We instantiate the DPG agent. We pass different options to it, such as the update
#    strategy, the optimizer, the Hessian type, etc. For plotting purposes, it is also
#    wrapped such that the updated parameters are recorded. And we also log the progress
#    of the simulation.
# 4. We run the simulation. Under the hood, the agent will interact with the
#    environment, collect data, and update the parameters of the MPC controller.
# 5. Finally, we plot the results. The first plot shows the evolution of the states and
#    the control action, and the corresponding bounds. The second plot shows the
#    performance per rollout, the norm of the estimated policy gradient per rollout, and
#    time-wise stage cost realizations. The last plot shows how each learnable parameter
#    evolves over time.

if __name__ == "__main__":
    # instantiate the env and wrap it - since we will train for only one long episode,
    # tell the DPG agent to perform its LSTD computations over subtrajectories of length
    # 100.
    env = MonitorEpisodes(TimeLimit(LtiSystem(), max_episode_steps=experiment_config["max_episode_steps"]))
    rollout_length = 100

    # now build the MPC and the dict of learnable parameters
    mpc = LinearMpc()
    learnable_pars = LearnableParametersDict[cs.SX](
        (
            LearnableParameter(name, val.shape, val, sym=mpc.parameters[name])
            for name, val in mpc.learnable_pars_init.items()
        )
    )

    # build and wrap appropriately the agent
    agent = Log(
        RecordUpdates(
            LstdDpgAgent(
                mpc=mpc,
                learnable_parameters=learnable_pars,
                discount_factor=mpc.discount_factor,
                optimizer=GradientDescent(learning_rate=experiment_config["learning_rate"]),
                update_strategy=UpdateStrategy(rollout_length, "on_timestep_end"),
                rollout_length=rollout_length,
                exploration=E.OrnsteinUhlenbeckExploration(0.0, 0.05*LtiSystem.a_bnd[1], mode="additive"),
                record_policy_performance=True,
                record_policy_gradient=True,
                use_last_action_on_fail=True,
            )
        ),
        level=logging.DEBUG,
        log_frequencies={"on_timestep_end": 1000},
    )

    # launch the training simulation
    agent.train(env=env, episodes=1, seed=69)

    # %%
    # Display the results
    # ----------------
    import matplotlib.pyplot as plt
    plt.rcParams['axes.xmargin'] = 0  # tight x range

    X = env.get_wrapper_attr("observations")[0].squeeze().T
    U = env.get_wrapper_attr("actions")[0].squeeze()
    R = env.get_wrapper_attr("rewards")[0]

    # scale the logs back to the physical values if needed
    if dimensionless:
        X = Mx @ X
        U = (Mu * U).ravel()
        # TODO: reward scaling back?

    vehicle_params = VehicleParams(vehicle_size=vehicle_size)
    isw = vehicle_params.isw

    e1 = X[0,:]
    e2 = X[2,:]
    delta_sw = isw * np.rad2deg(U)
    x = range(len(e1))
    steer_max = np.rad2deg(vehicle_params.sw_max)
    dsteer_max = steer_max  # max. steering rate in deg/s
    ddelta_sw = 1/LinearMpc.dt * np.diff(delta_sw, prepend=0)  # steering rate in deg/s

    # results in the error frame
    fig1, axs1 = plt.subplots(6, sharex=True, constrained_layout=True)
    fig1.suptitle('Vehicle steering in closed loop (error frame)')
    axs1[0].plot(x, e1)
    axs1[0].set_ylabel('$e_y$ [m]')
    axs1[1].plot(x, np.rad2deg(e2))
    axs1[1].set_ylabel(r'$e_\psi$ [deg]')
    axs1[2].plot(x, X[1,:])
    axs1[2].set_ylabel(r'$\dot{e}_y$ [m/s]')
    axs1[3].plot(x, np.rad2deg(X[3,:]))
    axs1[3].set_ylabel(r'$\dot{e}_\psi$ [deg/s]')
    axs1[4].plot(x[:-1], delta_sw)
    axs1[4].axhline( steer_max, color="r", linestyle='--')
    axs1[4].axhline(-steer_max, color="r", linestyle='--')
    axs1[4].set_ylabel(r'$\delta_\mathrm{sw}$ [deg]')
    axs1[5].plot(x[:-1], ddelta_sw)
    axs1[5].axhline( dsteer_max, color="r", linestyle='--')
    axs1[5].axhline(-dsteer_max, color="r", linestyle='--')
    axs1[5].set_ylabel(r'$\dot{\delta}_\mathrm{sw}$ [deg/s]')
    axs1[5].set_xlabel('$k$')
    fig1.align_ylabels()

    # # results in the inertial frame
    # _, y_ref, psi_ref = get_double_lane_change_data(x)
    # _, y = frenet2inertial(e1=e1, e2=e2, psi_ref=psi_ref, vx=env.vx, dt=env.dt)
    # psi = psi_ref + e2

    # fig, ax = plt.subplots(3, sharex=True, constrained_layout=True)
    # fig.suptitle('Vehicle steering in closed loop (inertial frame)')

    # ax[0].plot(x, y)
    # ax[0].plot(x, y_ref, 'r--')    
    # ax[0].set_ylabel('$y$ [m]')

    # ax[1].plot(x, np.rad2deg(psi))
    # ax[1].plot(x, np.rad2deg(psi_ref), 'r--')
    # ax[1].set_ylabel('$\psi$ [deg]')

    # ax[2].plot(x, delta_sw)
    # ax[2].axhline( steer_max, color="r", linestyle='--')
    # ax[2].axhline(-steer_max, color="r", linestyle='--')
    # ax[2].set_ylabel('$\delta_\mathrm{sw}$ [deg]')
    # ax[2].set_xlabel('$x$ [m]')


    # _, axs = plt.subplots(3, 1, constrained_layout=True, sharex=True)
    # axs[0].plot(X[0])
    # axs[1].plot(X[1])
    # axs[2].plot(U)
    # for i in range(2):
    #     axs[0].axhline(env.get_wrapper_attr("x_bnd")[i][0], color="r")
    #     axs[1].axhline(env.get_wrapper_attr("x_bnd")[i][1], color="r")
    #     axs[2].axhline(env.get_wrapper_attr("a_bnd")[i], color="r")
    # axs[0].set_ylabel("$s_1$")
    # axs[1].set_ylabel("$s_2$")
    # axs[2].set_ylabel("$a$")

    fig2, axs2 = plt.subplots(3, 1, constrained_layout=True)
    fig2.suptitle('Performance and policy gradient')
    axs2[0].plot(agent.policy_performances)
    axs2[1].semilogy(np.linalg.norm(agent.policy_gradients, axis=1))
    axs2[2].semilogy(R, "o", markersize=1)
    axs2[0].set_ylabel(r"$J(\pi_\theta)$")
    axs2[1].set_ylabel(r"$||\nabla_\theta J(\pi_\theta)||$")
    axs2[2].set_ylabel("$L$")
    fig2.align_ylabels()

    fig3, axs3 = plt.subplots(3, 2, constrained_layout=True, sharex=True)
    fig3.suptitle('Parameter values')
    axs3[0, 0].plot(np.asarray(agent.updates_history["b"]))
    axs3[0, 1].plot(
        np.stack(
            [np.asarray(agent.updates_history[n])[:, 0] for n in ("x_lb", "x_ub")], -1
        ),
    )
    axs3[1, 0].plot(np.asarray(agent.updates_history["f"]))
    axs3[1, 1].plot(np.asarray(agent.updates_history["V0"]))
    axs3[2, 0].plot(np.asarray(agent.updates_history["A"]).reshape(-1, 16))
    axs3[2, 1].plot(np.asarray(agent.updates_history["B"]).squeeze())
    axs3[0, 0].set_ylabel("$b$")
    axs3[0, 1].set_ylabel("$x$ backoff")
    axs3[1, 0].set_ylabel("$f$")
    axs3[1, 1].set_ylabel("$V_0$")
    axs3[2, 0].set_ylabel("$A$")
    axs3[2, 1].set_ylabel("$B$")
    fig3.align_ylabels()

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
