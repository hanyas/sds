import gym
from gym import spaces
from gym.utils import seeding

import numpy as np


def cart2ang(x):
    if x.ndim == 1:
        state = np.zeros((2,))
        state[0] = np.arctan2(x[1], x[0])
        state[1] = x[2]
        return state
    else:
        return np.vstack(list(map(cart2ang, list(x))))


def ang2cart(x):
    if x.ndim == 1:
        state = np.zeros((3,))
        state[0] = np.cos(x[0])
        state[1] = np.sin(x[0])
        state[2] = x[1]
        return state
    else:
        return np.vstack(list(map(ang2cart, list(x))))


class HybridPendulum(gym.Env):

    def __init__(self, rarhmm):
        self.state_dim = 2
        self.act_dim = 1
        self.obs_dim = 2

        # g = [th, thd]
        self.g = np.array([0., 0.])
        self.gw = - np.array([1e0, 1e-1])

        # x = [th, thd]
        self._state_max = np.array([np.inf, 8.0])

        # x = [th, thd]
        self.xmax = np.array([np.inf, np.inf])
        self.state_space = spaces.Box(low=-self.xmax,
                                      high=self.xmax,
                                      dtype=np.float64)

        # y = [th, thd]
        self.ymax = np.array([np.inf, np.inf])
        self.observation_space = spaces.Box(low=-self.ymax,
                                            high=self.ymax,
                                            dtype=np.float64)

        self.uw = - 1e-3 * np.ones((self.act_dim, ))
        self.umax = 2.5 * np.ones((self.act_dim, ))
        self.action_space = spaces.Box(low=-self.umax,
                                       high=self.umax,
                                       dtype=np.float64)

        self.rarhmm = rarhmm

        self.obs = None
        self.hist_obs = np.empty((0, self.obs_dim))
        self.hist_act = np.empty((0, self.act_dim))

        self.np_random = None

        self.seed()

    @property
    def xlim(self):
        return self.xmax

    @property
    def ulim(self):
        return self.umax

    def dynamics(self, xhist, uhist):
        xhist, uhist = np.atleast_2d(xhist, uhist)
        xn = self.rarhmm.step(xhist, uhist, stoch=False, average=True)
        return xn

    def rewrad(self, x, u):
        _x = cart2ang(x)
        return (_x - self.g).T @ np.diag(self.gw) @ (_x - self.g)\
               + u.T @ np.diag(self.uw) @ u

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, act):
        self.hist_act = np.vstack((self.hist_act, act))
        self.obs = self.dynamics(self.hist_obs, self.hist_act)
        self.hist_obs = np.vstack((self.hist_obs, self.obs))
        return self.obs, None, False, {}

    def reset(self):
        pass

    # for plotting
    def fake_step(self, obs, act):
        nxt_obs = self.dynamics(obs, act)
        return nxt_obs


class HybridPendulumWithCartesianObservation(HybridPendulum):

    def __init__(self, rarhmm):
        super(HybridPendulumWithCartesianObservation, self).__init__(rarhmm)
        self.obs_dim = 3

        # y = [cos, sin, thd]
        self.ymax = np.array([1., 1., np.inf])
        self.observation_space = spaces.Box(low=-self.ymax,
                                            high=self.ymax,
                                            dtype=np.float64)

    # for plotting
    def fake_step(self, obs, act):
        query = ang2cart(obs)
        nxt_obs = self.dynamics(query, act)
        return cart2ang(nxt_obs)
