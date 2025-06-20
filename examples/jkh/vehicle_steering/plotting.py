import matplotlib.pyplot as plt
from contextlib import contextmanager
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

plt.rcParams['axes.xmargin'] = 0  # tight x range

@contextmanager
def try_plot():
    try:
        yield
    except Exception as e:
        print(f"Plot skipped: {e}")


def plot_trajectory_error_frame(X, U, vehicle_params):
    e1 = X[0,:]
    e2 = X[2,:]
    delta_sw = vehicle_params["isw"] * np.rad2deg(U)
    x = range(len(e1))
    steer_max = np.rad2deg(vehicle_params["sw_max"])
    dsteer_max = steer_max  # max. steering rate in deg/s
    ddelta_sw = 1/vehicle_params["dt"] * np.diff(delta_sw, prepend=0)  # steering rate in deg/s

    # results in the error frame
    try:
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
    except:
        print("Could not plot all error frame results.")

    return fig1


def plot_performance(agent, R):
    try:
        fig2, axs2 = plt.subplots(3, 1, constrained_layout=True)
        fig2.suptitle('Performance and policy gradient')
        axs2[0].plot(agent.policy_performances)
        axs2[0].set_ylabel(r"$J(\pi_\theta)$")

        axs2[1].semilogy(np.linalg.norm(agent.policy_gradients, axis=1))
        axs2[1].set_ylabel(r"$||\nabla_\theta J(\pi_\theta)||$")
        
        axs2[2].semilogy(R, "o", markersize=1)
        axs2[2].set_ylabel("$L$")

        fig2.align_ylabels()
    except:
        print("Could not plot both the performance and policy gradient.")


def plot_parameters(agent):
    fig3, axs3 = plt.subplots(3, 2, constrained_layout=True, sharex=True)
    fig3.suptitle('Parameter values')
    
    with try_plot():
        axs3[0, 0].plot(np.asarray(agent.updates_history["b"]))
        axs3[0, 0].set_ylabel("$b$")
    
    with try_plot():
        axs3[0, 1].plot(
            np.stack(
                [np.asarray(agent.updates_history[n])[:, 0] for n in ("x_lb", "x_ub")], -1
            ),
        )
        axs3[0, 1].set_ylabel("$x$ backoff")

    with try_plot():
        axs3[1, 0].plot(np.asarray(agent.updates_history["f"]))
        axs3[1, 0].set_ylabel("$f$")

    # with try_plot():
    #     axs3[1, 1].plot(np.asarray(agent.updates_history["V0"]))
    #     axs3[1, 1].set_ylabel("$V_0$")

    # with try_plot():    
    #     axs3[2, 0].plot(np.asarray(agent.updates_history["A"]).reshape(-1, 16))
    #     axs3[2, 0].set_ylabel("$A$")
    
    # with try_plot():
    #     axs3[2, 1].plot(np.asarray(agent.updates_history["B"]).squeeze())
    #     axs3[2, 1].set_ylabel("$B$")

    fig3.align_ylabels()

    return fig3

def plot_trajectories(trajectories):
    N = len(trajectories)
    cmap = plt.get_cmap('Blues')
    norm = mcolors.Normalize(vmin=1, vmax=N)

    fig4, axs4 = plt.subplots(4, sharex=True, constrained_layout=True)
    fig4.suptitle('Learning trajectories')
    state_labels = ['$e_y$ [m]', r'$e_\psi$ [deg]', r'$\dot{e}_y$ [m/s]', r'$\dot{e}_\psi$ [deg/s]']
    for i, trajectory in enumerate(trajectories, start=1):
        traj = np.asarray(trajectory)
        for k in range(4):
            axs4[k].plot(traj[:,k,:], color=cmap(norm(i)))
    for k in range(4):
        axs4[k].set_ylabel(state_labels[k])
    axs4[3].set_xlabel('$k$')

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig4.colorbar(sm, ax=axs4)
    cbar.set_ticks([1, N])
    cbar.set_ticklabels(["1", f"{N}"])

    fig4.align_ylabels()

    return fig4


def plot_trajectory_inertial_frame(X, U, vehicle_params):
    raise NotImplementedError("Plotting in inertial frame is not implemented yet.")
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