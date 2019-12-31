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
    import rl

    # np.random.seed(1337)

    env = gym.make('Cartpole-ID-v0')
    # env = gym.make('HybridPendulum-ID-v0')
    env._max_episode_steps = 5000
    # env.seed(1337)

    xlim = (-0., 0.)
    thlim = (-np.pi, np.pi)
    xdlim = (-0., 0.)
    thdlim = (-10.0, 10.0)

    npts = 25

    x = np.linspace(*xlim, npts)
    th = np.linspace(*thlim, npts)
    xd = np.linspace(*xdlim, npts)
    thd = np.linspace(*thdlim, npts)

    X, TH, Xd, THd = np.meshgrid(x, th, xd, thd, indexing='ij')
    S = np.stack((X, TH, Xd, THd))

    Sn = np.zeros((4, npts, npts, npts, npts))
    Zn = np.zeros((npts, npts, npts, npts))

    env.reset()
    for i in range(1):
        for j in range(npts):
            for k in range(1):
                for l in range(npts):
                    Sn[:, i, j, k, l] = env.unwrapped.fake_step(S[:, i, j, k, l], np.array([0.0]))
                    # Zn[i, j], XYn[:, i, j] = env.unwrapped.fake_step(XY[:, i, j], np.array([0.0]))

    dydt = Sn - S

    fig = plt.figure(figsize=(5, 5), frameon=True)
    ax = fig.gca()

    ax.streamplot(th, thd, dydt[1, 0, :, 0, :], dydt[3, 0, :, 0, :],
                  color='b', linewidth=1, density=1.25,
                  arrowstyle='->', arrowsize=1.5)

    ax = beautify(ax)
    ax.grid(False)

    ax.set_xlim(thlim)
    ax.set_ylim(thdlim)

    # from tikzplotlib import save
    # save("cartpole_openloop.tex")

    plt.show()
