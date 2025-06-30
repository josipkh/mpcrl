import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from utils import get_double_lane_change_data, frenet2inertial

plt.rcParams['axes.xmargin'] = 0  # tight x range


def plot_trajectory_error_frame(X, U, vehicle_params):
    """A single trajectory in the error frame, with the input and input rate."""
    e1 = X[0,:]
    e2 = X[2,:]
    delta_sw = vehicle_params["isw"] * np.rad2deg(U)
    x = vehicle_params["vx"] * vehicle_params["dt"] * np.asarray(list(range(e1.shape[0])))
    steer_max = np.rad2deg(vehicle_params["sw_max"])
    dsteer_max = steer_max  # max. steering rate in deg/s
    ddelta_sw = 1/vehicle_params["dt"] * np.diff(delta_sw, prepend=0)  # steering rate in deg/s

    # results in the error frame
    try:
        fig, axs = plt.subplots(6, sharex=True, constrained_layout=True)
        fig.suptitle('Vehicle steering in closed loop (error frame)')
        axs[0].plot(x, e1)
        axs[0].set_ylabel('$e_y$ [m]')

        axs[1].plot(x, X[1,:])
        axs[1].set_ylabel(r'$\dot{e}_y$ [m/s]')

        axs[2].plot(x, np.rad2deg(e2))
        axs[2].set_ylabel(r'$e_\psi$ [deg]')

        axs[3].plot(x, np.rad2deg(X[3,:]))
        axs[3].set_ylabel(r'$\dot{e}_\psi$ [deg/s]')

        axs[4].step(x, delta_sw, where="post")
        axs[4].axhline( steer_max, color="r", linestyle='--')
        axs[4].axhline(-steer_max, color="r", linestyle='--')
        axs[4].set_ylabel(r'$\delta_\mathrm{sw}$ [deg]')

        axs[5].step(x, ddelta_sw, where="post")
        axs[5].axhline( dsteer_max, color="r", linestyle='--')
        axs[5].axhline(-dsteer_max, color="r", linestyle='--')
        axs[5].set_ylabel(r'$\dot{\delta}_\mathrm{sw}$ [deg/s]')

        axs[5].set_xlabel('$x$ [m]')
        fig.align_ylabels()
    except:
        print("Could not plot all error frame results.")

    return fig


def plot_performance(agent, R):
    """Policy gradients and cost."""
    try:
        fig, axs = plt.subplots(3, 1, constrained_layout=True)
        fig.suptitle('Performance and policy gradient')
        axs[0].plot(agent.policy_performances)
        axs[0].set_ylabel(r"$J(\pi_\theta)$")

        axs[1].semilogy(np.linalg.norm(agent.policy_gradients, axis=1))
        axs[1].set_ylabel(r"$||\nabla_\theta J(\pi_\theta)||$")
        
        axs[2].semilogy(R, "o", markersize=1)
        axs[2].set_ylabel("$L$")

        fig.align_ylabels()
    except:
        print("Could not plot both the performance and policy gradient.")


def plot_parameters(agent):
    """Parameter updates during learning."""
    n_learnable_params = len(agent.updates_history)
    n_rows = int(np.ceil(n_learnable_params / 2))

    fig, axs = plt.subplots(n_rows, 2, constrained_layout=True, sharex=True)
    axs = axs.flatten()
    fig.suptitle('Parameter values')

    for k, (key, values) in enumerate(agent.updates_history.items()):
        ax = axs[k]
        ax.ticklabel_format(useOffset=False)
        n_elements = values[0].shape[0] if len(values[0].shape) > 0 else 1
        is_vector = n_elements > 1
        if is_vector:
            for j in range(n_elements):
                ax.plot([value[j] for value in values], label=f'{key}_{j}')
        else:
            ax.plot(values)
        ax.set_ylabel("$"+key+"$")
        if is_vector:
            ax.legend()

    fig.align_ylabels()

    return fig


def plot_trajectories_error_frame(trajectories, vehicle_params):
    """System trajectories during learning, in the error frame."""
    N = len(trajectories)

    if N == 1:
        trajectory = np.asarray(trajectories[0]).squeeze()
        return plot_trajectory_error_frame(
            X=trajectory[:, 0:4].T, U=trajectory[:, 4].T, vehicle_params=vehicle_params
            )
    else:
        cmap = plt.get_cmap('Blues')
        norm = mcolors.Normalize(vmin=1, vmax=N)
        steer_max = np.rad2deg(vehicle_params["sw_max"])

        fig, axs = plt.subplots(5, sharex=True, constrained_layout=True)
        fig.suptitle('Learning trajectories')
        labels = ['$e_y$ [m]', r'$\dot{e}_y$ [m/s]', r'$e_\psi$ [deg]', r'$\dot{e}_\psi$ [deg/s]', r'$\delta_\mathrm{sw}$ [deg]']
        for i, trajectory in enumerate(trajectories, start=1):
            traj = np.asarray(trajectory)
            color = cmap(norm(i)) if N>1 else "#1f77b4"
            for k in range(len(labels)):
                if k < 4:
                    axs[k].plot(traj[:,k,:] if k < 2 else np.rad2deg(traj[:,k,:]), color=color)
                else:
                    axs[k].step(range(traj.shape[0]), vehicle_params["isw"]*np.rad2deg(traj[:,k,:]), color=color, where="post")
                    axs[k].axhline( steer_max, color="r", linestyle='--')
                    axs[k].axhline(-steer_max, color="r", linestyle='--')
        for k in range(len(labels)):
            axs[k].set_ylabel(labels[k])
        axs[-1].set_xlabel('$k$')

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axs)
        cbar.set_ticks([1, N])
        cbar.set_ticklabels(["1", f"{N}"])

        fig.align_ylabels()

        return fig


def plot_trajectory_inertial_frame(X, U, vehicle_params):
    """System trajectory in the inertial frame."""
    vx, dt = vehicle_params["vx"], vehicle_params["dt"]
    steer_max = np.rad2deg(vehicle_params["sw_max"])
    delta_sw = vehicle_params["isw"] * np.rad2deg(U)
    e1 = X[0,:]
    e2 = X[2,:]

    match vehicle_params["maneuver"]:
        case "double_lane_change":
            _, y_ref, yaw_ref = get_double_lane_change_data(X=0, horizon=X.shape[1], vehicle_params=vehicle_params)
        case "straight":
            y_ref = np.zeros_like(e1)
            yaw_ref = np.zeros_like(e2)
    _, y = frenet2inertial(e1=e1, e2=e2, yaw_ref=yaw_ref, vx=vx, dt=dt)
    yaw = yaw_ref + e2
    x = vx * dt * np.asarray(list(range(y.shape[0])))

    fig, ax = plt.subplots(3, sharex=True, constrained_layout=True)
    fig.suptitle('Vehicle steering in closed loop (inertial frame)')

    ax[0].plot(x, y)
    ax[0].plot(x, y_ref, 'r--')    
    ax[0].set_ylabel('$y$ [m]')

    ax[1].plot(x, np.rad2deg(yaw))
    ax[1].plot(x, np.rad2deg(yaw_ref), 'r--')
    ax[1].set_ylabel(r'$\psi$ [deg]')

    ax[2].step(x, delta_sw, where="post")
    ax[2].axhline( steer_max, color="r", linestyle='--')
    ax[2].axhline(-steer_max, color="r", linestyle='--')
    ax[2].set_ylabel(r'$\delta_\mathrm{sw}$ [deg]')
    ax[2].set_xlabel('$x$ [m]')

    fig.align_ylabels()

    return fig


def plot_trajectories_inertial_frame(trajectories, vehicle_params):
    """System trajectories during learning, in the inertial frame."""
    N = len(trajectories)

    if N == 1:
        trajectory = np.asarray(trajectories[0]).squeeze()
        return plot_trajectory_inertial_frame(
            X=trajectory[:, 0:4].T, U=trajectory[:, 4].T, vehicle_params=vehicle_params
            )
    else:
        vx, dt = vehicle_params["vx"], vehicle_params["dt"]
        cmap = plt.get_cmap('Blues')
        norm = mcolors.Normalize(vmin=1, vmax=N)
        steer_max = np.rad2deg(vehicle_params["sw_max"])

        fig, axs = plt.subplots(3, sharex=True, constrained_layout=True)
        fig.suptitle('Learning trajectories')
        labels = ['$Y$ [m]', r'$e_\psi$ [deg]', r'$\delta_\mathrm{sw}$ [deg]']

        # add the reference trajectory if needed
        if vehicle_params["maneuver"] == "double_lane_change":
            max_steps = max([len(trajectory) for trajectory in trajectories])
            x = vx * dt * np.asarray(list(range(max_steps)))
            _, y_ref, yaw_ref = get_double_lane_change_data(X=0, horizon=max_steps, vehicle_params=vehicle_params)
            axs[0].plot(x, y_ref, 'r--')
            axs[1].plot(x, np.rad2deg(yaw_ref), 'r--')
        
        # plot individual trajectories
        for i, trajectory in enumerate(trajectories, start=1):
            # convert trajectory to inertial frame
            traj = np.asarray(trajectory).squeeze().T
            x = vx * dt * np.asarray(list(range(traj.shape[1])))
            e1 = traj[0,:]
            e2 = traj[2,:]
            delta_sw = vehicle_params["isw"] * np.rad2deg(traj[4,:])
            match vehicle_params["maneuver"]:
                case "double_lane_change":
                    yaw_ref = get_double_lane_change_data(X=0, horizon=e1.shape[0], vehicle_params=vehicle_params)[2]
                case "straight":
                    yaw_ref = np.zeros_like(e2)
            _, y = frenet2inertial(e1=e1, e2=e2, yaw_ref=yaw_ref, vx=vx, dt=dt)
            yaw = yaw_ref + e2

            # plot the trajectory
            color = cmap(norm(i)) if N>1 else "#1f77b4"
            axs[0].plot(x, y, color=color)
            axs[1].plot(x, np.rad2deg(yaw), color=color)
            axs[2].step(x, delta_sw, color=color, where="post")
        
        axs[2].axhline( steer_max, color="r", linestyle='--')
        axs[2].axhline(-steer_max, color="r", linestyle='--')
        for k in range(len(labels)):
            axs[k].set_ylabel(labels[k])
        axs[-1].set_xlabel('$x$ [m]')

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axs)
        cbar.set_ticks([1, N])
        cbar.set_ticklabels(["1", f"{N}"])

        fig.align_ylabels()

        return fig
