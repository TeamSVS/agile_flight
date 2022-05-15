import functools
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from src.model.quadrotor_model_execution import control_trajectory, create_capsule, set_ocp, acados_solve, get_ocp_result



def plotlive(func):
    plt.ion()

    @functools.wraps(func)
    def new_func(*args, **kwargs):

        # Clear all axes in the current figure.
        axes = plt.gcf().get_axes()
        for axis in axes:
            axis.cla()

        # Call func to plot something
        result = func(*args, **kwargs)

        # Draw the plot
        plt.draw()
        plt.pause(0.01)

        return result

    return new_func 

def plot_solution(t, simX, simU, realX=None, save_fig=None):
    fig, ax = plt.subplots(nrows=6, ncols=1, sharex=True, figsize=(10, 8))
    i = 0
    for a, ls in zip(ax, ["x y z", "qw qx qy qz", "vx vy vz", "wx wy wz", "r1 r2 r3 r4", "ax az az"]):
        a.cla()
        for l, style in zip(ls.split(" "), ['-', '--', '-.', ':']):
            if i < 13:
                s = a.plot(t, simX[:,i], label=l)
                if realX is not None:
                    a.plot(t, realX[:,i], label=l+"_real", linestyle="--", color=s[0].get_color())
            elif i >= 13 and i <= 16:
                a.plot(t[:-1], simU[:,i-13], label=l)
            else:
                v = simX[:,7:10]
                acc = v[1:, :] - v[:-1, :]
                a.plot(t[:-1], acc[:, i-17], label=l)
            i += 1
        a.legend()
    ax[-1].set_xlabel("t [s]")

    fig.suptitle(f"Quadrotor Stabilization")
    # fig.savefig(f"quadrotor_{model_name}.pdf")
    plt.show(block=True)
    if save_fig is not None:
        fig.savefig(save_fig)


def test(params):
    setpoint = np.array([
        0, 2, 0,
        1, 0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0, 0
    ]).astype(float)
    traj = np.repeat(setpoint.reshape((1, -1)), params["N"]+1, axis=0)
    # v = value.copy()
    # traj[:, 9] = -3/(((np.arange(N+1)/N - 0.5)*4)**2 + 1)

    phi = 0/180 * np.pi
    x0 = np.array([
        0, 0, 0,
        np.cos(phi/2), *(np.sin(phi/2)*np.array([1, 0, 0])),
        0, 0, 0,
        0, 0, 0,
    ]).astype(float)
    print(f"x0 {x0}")
    capsule = create_capsule()
    t, simX, simU = control_trajectory(capsule, x0, traj, params["N"], params["Tf"])
    plot_solution(t, simX, simU)


def test_speed(params, num_tests=1000):
    # initial_orientation = R.random()
    initial_orientations = [R.from_euler('xyz', [np.random.random() * 10/180*np.pi, 0, 0]) for _ in range(num_tests)]
    final_orientations = map(lambda x: x * R.from_euler('xyz', [00/180*np.pi, 0, 0]), initial_orientations)
    initial_orientations = np.vstack(map(lambda x: np.array([x.as_quat()[3], *x.as_quat()[:3]]), initial_orientations))
    final_orientations = np.vstack(map(lambda x: np.array([x.as_quat()[3], *x.as_quat()[:3]]), final_orientations))


    capsules = [create_capsule() for _ in range(num_tests)]

    for i, (initial_orientation, final_orientation, capsule) in enumerate(zip(initial_orientations, final_orientations, capsules)):
        setpoint = np.array([
            0, 0, 0,
            # *final_orientation,
            1, 0, 0, 0,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0, 0
        ]).astype(float)
        traj = np.repeat(setpoint.reshape((1, -1)), params["N"]+1, axis=0)

        phi = 0/180 * np.pi
        x0 = np.array([
            0, 0, 0,
            *initial_orientation,
            # 1, 0, 0, 0,
            # np.cos(phi/2), *(np.sin(phi/2)*np.array([1, 0, 0])),
            0, 0, 0,
            0, 0, 0,
        ]).astype(float)
        set_ocp(capsule, x0, traj, params["N"], params["Tf"])

    start = time.time()
    for capsule in capsules:
        acados_solve(capsule["capsule"])
    end = time.time()
    if num_tests > 1:
        print(f"Solving {num_tests} times took {end - start} seconds")

    plot_solution(*get_ocp_result(capsules[-1], params["N"], params["Tf"]))


if __name__ == "__main__":
    from src.params import flightmare
    from ..model.quadrotor_model_generation import generate
    params = flightmare.params()
    
    generate(params)
    # test(params)
    test_speed(params, num_tests=100)