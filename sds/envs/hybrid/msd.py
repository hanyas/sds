import gym
from gym import spaces
from gym.utils import seeding

import autograd.numpy as np

from sds import rARHMM


def mass_spring_damper(param):
    k, m, d, s, const = param

    A = np.array([[0.0, 1.0], [- k / m, - d / m]])
    B = np.array([[0.0, 1.0 / m]]).T
    c = np.array([const, k * s / m])

    return A, B, c


class MassSpringDamper(gym.Env):

    def __init__(self):
        self.dm_state = 2
        self.dm_act = 1
        self.dm_obs = 2

        self._dt = 0.01

        self._goal = np.array([1., 0.])
        self._goal_weight = - np.array([1.e0, 1.e-1])

        self._state_max = np.array([10., 10.])

        self._obs_max = np.array([10., 10.])
        self.observation_space = spaces.Box(low=-self._obs_max,
                                            high=self._obs_max)

        self._act_weight = - np.array([1.e-3])
        self._act_max = 10.
        self.action_space = spaces.Box(low=-self._act_max,
                                       high=self._act_max, shape=(1,))

        # define the swithching dynamics
        # k, m, d, s, const
        param = ([0.5, 0.25, 0.25, -5.0, 0.0],
                 [-0.5, 0.25, 0.25, 5.0, 0.0])

        self.rarhmm = rARHMM(nb_states=2,
                             dm_obs=self.dm_obs,
                             dm_act=self.dm_act,
                             trans_type='poly')

        _sigma = np.zeros((2, self.dm_state, self.dm_state))
        _init_sigma = np.zeros((2, self.dm_state, self.dm_state))
        for k in range(2):
            A, B, c = mass_spring_damper(param[k])
            self.rarhmm.observations.A[k, ...] = np.eye(self.dm_obs) + self._dt * A
            self.rarhmm.observations.B[k, ...] = self._dt * B
            self.rarhmm.observations.c[k, ...] = self._dt * c
            _sigma[k, ...] = 1.e-4 * np.eye(self.dm_state)

            self.rarhmm.init_observation.mu[k, :] = np.zeros((self.dm_obs, ))
            _init_sigma[k, ...] = 1e-4 * np.eye(self.dm_obs)

        self.rarhmm.observations.cov = _sigma
        self.rarhmm.init_observation.cov = _init_sigma

        self.rarhmm.transitions.coef = np.array([[1.0, 1.0, 0.0],
                                                 [5.0, 5.0, 0.0]])

        self.obs = None

        self.hist_obs = np.empty((0, self.dm_obs))
        self.hist_act = np.empty((0, self.dm_act))

        self.np_random = None

        self.seed()

    @property
    def xlim(self):
        return self._state_max

    @property
    def ulim(self):
        return self._act_max

    @property
    def dt(self):
        return self._dt

    @property
    def goal(self):
        return self._goal

    def dynamics(self, xhist, uhist):
        xhist = np.atleast_2d(xhist)
        uhist = np.atleast_2d(uhist)

        # filter hidden state
        b = self.rarhmm.filter(xhist, uhist)[0][-1, ...]

        # evolve dynamics
        x, u = xhist[-1, :], uhist[-1, :]
        zn, xn = self.rarhmm.step(x, u, b, stoch=False, mix=False)

        return zn, xn

    def rewrad(self, x, u):
        return (x - self._goal).T @ np.diag(self._goal_weight) @ (x - self._goal)\
               + u.T @ np.diag(self._act_weight) @ u

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, act):
        # apply action constraints
        _act = np.clip(act, -self._act_max, self._act_max)
        self.hist_act = np.vstack((self.hist_act, _act))

        # compute reward
        rwrd = self.rewrad(self.obs, _act)

        # evolve dynamics
        _, self.obs = self.dynamics(self.hist_obs, self.hist_act)
        self.hist_obs = np.vstack((self.hist_obs, self.obs))

        return self.obs, rwrd, False, {}

    def reset(self):
        self.hist_obs = np.empty((0, self.dm_obs))
        self.hist_act = np.empty((0, self.dm_act))

        _state = self.rarhmm.init_state.sample()

        self.obs = self.rarhmm.init_observation.sample(z=_state)
        self.hist_obs = np.vstack((self.hist_obs, self.obs))

        return self.obs

    # following function for plotting
    def fake_step(self, value, act):
        # switch to observation space
        _obs = value

        # apply action constraints
        _act = np.clip(act, -self._act_max, self._act_max)

        # evolve dynamics
        _nxt_state, _nxt_obs = self.dynamics(_obs, _act)

        return _nxt_state, _nxt_obs
