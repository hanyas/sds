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
                u = npr.uniform(-ulim, ulim)
            else:
                u = ctl(x) + noise_std * npr.randn(1, )

            if apply_limit:
                u = np.clip(u, -ulim, ulim)

            _obs[t, :], _act[t, :] = x, u
            x, r, _, _ = env.step(u)

        obs.append(_obs)
        act.append(_act)

    return obs, act


def rollout_policy(env, model, nb_rollouts, nb_steps,
                   average=False, stoch=False):

    rollouts = []

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    ulim = env.action_space.high
    nb_states = model.nb_states

    for n in range(nb_rollouts):
        roll = {'z': np.empty((0,), np.int64),
                'b': np.empty((0, nb_states)),  # belief
                'x': np.empty((0, obs_dim)),
                'u': np.empty((0, act_dim)),
                'r': np.empty((0, ))}

        x = env.reset()
        roll['x'] = np.vstack((roll['x'], x))

        b = model.init_state.pi
        roll['b'] = np.vstack((roll['b'], b))

        z = npr.choice(nb_states, p=b) if stoch else np.argmax(b)
        roll['z'] = np.hstack((roll['z'], z))

        u = np.zeros((act_dim, ))
        if average:
            for k in range(nb_states):
                u += b[k] * model.controls.sample(k, x) if stoch \
                    else model.controls.mean(k, x)
        else:
            u = model.controls.sample(z, x) if stoch \
                else model.controls.mean(z, x)

        u = np.clip(u, -ulim, ulim)
        roll['u'] = np.vstack((roll['u'], u))

        for t in range(nb_steps):
            x, r, _, _ = env.step(u)
            roll['x'] = np.vstack((roll['x'], x))
            roll['r'] = np.hstack((roll['r'], r))

            # pad action for filter
            aug_u = np.vstack((roll['u'], np.zeros((1, act_dim))))

            b = model.filtered_state(roll['x'], aug_u)[-1]
            roll['b'] = np.vstack((roll['b'], b))

            z = npr.choice(nb_states, p=b) if stoch else np.argmax(b)
            roll['z'] = np.hstack((roll['z'], z))

            u = np.zeros((act_dim, ))
            if average:
                for k in range(nb_states):
                    u += b[k] * model.controls.sample(k, x) if stoch\
                         else model.controls.mean(k, x)
            else:
                u = model.controls.sample(z, x) if stoch\
                    else model.controls.mean(z, x)

            u = np.clip(u, -ulim, ulim)
            roll['u'] = np.vstack((roll['u'], u))

        rollouts.append(roll)
    return rollouts


def rollout_ensemble(env, ensemble, nb_rollouts, nb_steps,
                     average=False, stoch=False):

    rollouts = []

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    ulim = env.action_space.high
    nb_states = ensemble.nb_states
    ensemble_size = ensemble.ensemble_size

    for n in range(nb_rollouts):
        roll = {'x': np.empty((0, obs_dim)),
                'u': np.empty((0, act_dim)),
                'r': np.empty((0, ))}

        x = env.reset()
        roll['x'] = np.vstack((roll['x'], x))

        us = np.zeros((ensemble_size, act_dim))
        for i, m in enumerate(ensemble.models):
            b = m.init_state.pi
            z = npr.choice(nb_states, p=b) if stoch else np.argmax(b)

            if average:
                for k in range(nb_states):
                    us[i] += b[k] * m.controls.sample(k, x) if stoch \
                        else m.controls.mean(k, x)
            else:
                us[i] = m.controls.sample(z, x) if stoch \
                    else m.controls.mean(z, x)

        u = np.mean(us, axis=0)
        u = np.clip(u, -ulim, ulim)
        roll['u'] = np.vstack((roll['u'], u))

        for t in range(nb_steps):
            x, r, _, _ = env.step(u)
            roll['x'] = np.vstack((roll['x'], x))
            roll['r'] = np.hstack((roll['r'], r))

            # pad action for filter
            aug_u = np.vstack((roll['u'], np.zeros((1, act_dim))))

            us = np.zeros((ensemble_size, act_dim))
            for i, m in enumerate(ensemble.models):
                b = m.filtered_state(roll['x'], aug_u)[-1]
                z = npr.choice(nb_states, p=b) if stoch else np.argmax(b)

                if average:
                    for k in range(nb_states):
                        us[i] += b[k] * m.controls.sample(k, x) if stoch \
                            else m.controls.mean(k, x)
                else:
                    us[i] = m.controls.sample(z, x) if stoch \
                        else m.controls.mean(z, x)

            u = np.mean(us, axis=0)
            u = np.clip(u, -ulim, ulim)
            roll['u'] = np.vstack((roll['u'], u))

        rollouts.append(roll)
    return rollouts
