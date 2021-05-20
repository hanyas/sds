import numpy as np
from numpy import random as npr


def brownian(x0, n, dt, delta, out=None):

    x0 = np.asarray(x0)

    # For each element of x0, generate a sample of n numbers from a
    # normal distribution.
    r = npr.randn(x0.shape[0], n) * delta * np.sqrt(dt)

    # If `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)

    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples.
    np.cumsum(r, axis=-1, out=out)

    # Add the initial condition.
    out += np.expand_dims(x0, axis=-1)

    return np.reshape(out, x0.shape[0])


def sample_env(env, nb_rollouts, nb_steps,
               ctl=None, noise_std=0.1,
               apply_limit=False):
    obs, act = [], []

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    ulim = env.action_space.high

    for n in range(nb_rollouts):
        _obs = np.zeros((nb_steps, obs_dim))
        _act = np.zeros((nb_steps, act_dim))

        x = env.reset()

        for t in range(nb_steps):
            if ctl is None:
                # unifrom distribution
                u = np.random.uniform(-ulim, ulim)
            else:
                u = ctl(x)
                u = u + noise_std * npr.randn(1, )

            if apply_limit:
                u = np.clip(u, -ulim, ulim)

            _obs[t, :] = x
            _act[t, :] = u

            x, r, _, _ = env.step(u)

        obs.append(_obs)
        act.append(_act)

    return obs, act
