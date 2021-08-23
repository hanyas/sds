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
    import sds

    true_env = gym.make('Pendulum-ID-v1')
    true_env._max_episode_steps = 5000

    hybrid_env = gym.make('HybridPendulum-ID-v1')
    hybrid_env._max_episode_steps = 5000

    xlim = (-np.pi, np.pi)
    ylim = (-8.0, 8.0)

    npts, hr = 36, 2

    x = np.linspace(*xlim, npts)
    y = np.linspace(*ylim, npts)

    X, Y = np.meshgrid(x, y)
    XYi = np.stack((X, Y))
    XYh = np.zeros((hr, 2, npts, npts))
    XYn = np.zeros((2, npts, npts))

    ##
    true_env.reset()
    for i in range(npts):
        for j in range(npts):
            XYh[0, :, i, j] = XYi[:, i, j]
            for t in range(1, hr):
                XYh[t, :, i, j] = true_env.unwrapped.fake_step(XYh[t - 1, :, i, j], np.array([0.0]))

    dXY = XYh[1, :, :, :] - XYi

    fig = plt.figure(figsize=(5, 5), frameon=True)
    ax = fig.gca()

    ax.streamplot(XYi[0, 1, :], XYi[1, :, 0], dXY[0, ...], dXY[1, ...],
                  color='b', linewidth=1, density=1.25,
                  arrowstyle='-|>', arrowsize=1.,
                  minlength=0.25)

    ax = beautify(ax)
    ax.grid(False)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    plt.show()

    ##
    hybrid_env.reset()
    for i in range(npts):
        for j in range(npts):
            Uh = np.zeros((hr, 1))
            XYn[:, i, j] = hybrid_env.unwrapped.fake_step(XYh[..., i, j], Uh)
    dXY = XYn - XYh[-1, ...]

    fig = plt.figure(figsize=(5, 5), frameon=True)
    ax = fig.gca()

    # re-interpolate data for streamplot
    xh, yh = XYh[-1, 0, 1, :], XYh[-1, 1, :, 0]
    xi = np.linspace(xh.min(), xh.max(), x.size)
    yi = np.linspace(yh.min(), yh.max(), y.size)

    from scipy.interpolate import interp2d

    dxh, dyh = dXY[0, ...], dXY[1, ...]
    dxi = interp2d(xh, yh, dxh)(xi, yi)
    dyi = interp2d(xh, yh, dyh)(xi, yi)

    ax.streamplot(xi, yi, dxi, dyi,
                  color='r', linewidth=1, density=1.25,
                  arrowstyle='-|>', arrowsize=1.,
                  minlength=0.25)

    ax = beautify(ax)
    ax.grid(False)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    from tikzplotlib import save
    save("pendulum_openloop.tex")

    plt.show()
