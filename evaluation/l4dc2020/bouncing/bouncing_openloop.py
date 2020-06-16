import numpy as np
import numpy.ma as ma

from matplotlib import rc
import matplotlib.pyplot as plt


rc('lines', **{'linewidth': 1})
rc('text', usetex=True)


def beautify(ax):
    ax.set_frame_on(True)
    ax.minorticks_on()

    ax.grid(True)
    ax.grid(linestyle=':')

    ax.tick_params(which='both', direction='in',
                   bottom=True, labelbottom=True,
                   top=True, labeltop=False,
                   right=True, labelright=False,
                   left=True, labelleft=True)

    ax.tick_params(which='major', length=6)
    ax.tick_params(which='minor', length=3)

    ax.autoscale(tight=True)
    # ax.set_aspect('equal')

    if ax.get_legend():
        ax.legend(loc='best')

    return ax


if __name__ == "__main__":

    import gym
    import sds_numpy

    # np.random.seed(1337)

    env = gym.make('BouncingBall-ID-v0')
    # env = gym.make('HybridBouncingBall-ID-v0')
    env._max_episode_steps = 5000
    # env.seed(1337)

    xlim = (-5., 10.)
    ylim = (-10.0, 10.0)

    npts = 26

    x = np.linspace(*xlim, npts)
    y = np.linspace(*ylim, npts)

    X, Y = np.meshgrid(x, y)
    XY = np.stack((X, Y))

    XYn = np.zeros((2, npts, npts))
    Zn = np.zeros((npts, npts))

    env.reset()
    for i in range(npts):
        for j in range(npts):
            XYn[:, i, j] = env.unwrapped.fake_step(XY[:, i, j], np.array([0.0]))
            # Zn[i, j], XYn[:, i, j] = env.unwrapped.fake_step(XY[:, i, j], np.array([0.0]))

    dydt = XYn - XY

    fig = plt.figure(figsize=(5, 5), frameon=True)
    ax = fig.gca()

    ax.streamplot(x, y, dydt[0, ...], dydt[1, ...],
                  color='b', linewidth=1, density=1.5,
                  arrowstyle='->', arrowsize=1.25)

    ax = beautify(ax)
    ax.grid(False)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # from tikzplotlib import save
    # save("bouncing_openloop.tex")

    plt.show()
