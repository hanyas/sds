import autograd.numpy as np
import autograd.numpy.random as npr

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
    ax.set_aspect('equal')

    if ax.get_legend():
        ax.legend(loc='best')

    return ax


def sample_env(env, nb_rollouts, nb_steps, ctl=None, max_act=10.):
    obs, act = [], []

    dm_obs = env.observation_space.shape[0]
    dm_act = env.action_space.shape[0]

    for n in range(nb_rollouts):
        _obs = np.empty((nb_steps, dm_obs))
        _act = np.empty((nb_steps, dm_act))

        x = env.reset()

        for t in range(nb_steps):
            if ctl is None:
                u = 2. * max_act * npr.randn(1, )
            else:
                u = ctl.actions(x, stoch=True)

            _obs[t, :] = x
            _act[t, :] = u

            x, r, _, _ = env.step(u)

        obs.append(_obs)
        act.append(_act)

    return obs, act


if __name__ == "__main__":

    import os
    import pickle

    import gym
    import rl

    from sds import rARHMM
    from sds import erARHMM

    env = gym.make('MassSpringDamper-ID-v0')
    env._max_episode_steps = 5000

    nb_rollouts, nb_steps = 10, 500
    dm_obs = env.observation_space.shape[0]
    dm_act = env.action_space.shape[0]
    nb_states = 2

    obs, act = sample_env(env, nb_rollouts, nb_steps)

    rarhmm = rARHMM(nb_states=nb_states,
                    dm_obs=dm_obs,
                    dm_act=dm_act,
                    type='recurrent')

    rarhmm.initialize(obs, act)
    lls = rarhmm.em(obs=obs, act=act, nb_iter=50, prec=1e-4, verbose=True)

    path = os.path.dirname(rl.__file__)
    pickle.dump(rarhmm, open(path + '/envs/control/hybrid/models/poly_rarhmm_msd.pkl', 'wb'))
