import gym
from gym import spaces
from gym.utils import seeding

import autograd.numpy as np


def end2ang(x):
    _state = np.zeros((2, ))
    _state[1] = x[2]
    _state[0] = np.arctan2(x[1], x[0])
    return _state


class HybridPendulum(gym.Env):

    def __init__(self, rarhmm):
        self.dm_state = 2
        self.dm_act = 1
        self.dm_obs = 2

        # g = [th, thd]
        self._goal = np.array([0., 0.])
        self._goal_weight = - np.array([1.e0, 1.e-1])

        # x = [th, thd]
        self._state_max = np.array([np.inf, 8.0])

        # o = [cos, sin, thd]
        self._obs_max = np.array([np.inf, 8.0])
        self.observation_space = spaces.Box(low=-self._obs_max,
                                            high=self._obs_max)

        self._act_weight = - np.array([1.e-3])
        self._act_max = 2.5
        self.action_space = spaces.Box(low=-self._act_max,
                                       high=self._act_max, shape=(1,))

        rarhmm.learn_ctl = False
        self.rarhmm = rarhmm

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
        _x = end2ang(x)
        return (_x - self._goal).T @ np.diag(self._goal_weight) @ (_x - self._goal)\
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


class HybridPendulumWithCartesianObservation(HybridPendulum):

    def __init__(self, rarhmm):
        super(HybridPendulumWithCartesianObservation, self).__init__(rarhmm)
        self.dm_obs = 3

        # o = [cos, sin, thd]
        self._obs_max = np.array([1., 1., 8.0])
        self.observation_space = spaces.Box(low=-self._obs_max,
                                            high=self._obs_max)

    def observe(self, x):
        return np.array([np.cos(x[0]),
                         np.sin(x[0]),
                         x[1]])

    # following function for plotting
    def fake_step(self, value, act):
        # switch to observation space
        _obs = self.observe(value)

        # apply action constraints
        _act = np.clip(act, -self._act_max, self._act_max)

        # evolve dynamics
        _nxt_state, _nxt_obs = self.dynamics(_obs, _act)

        return _nxt_state, end2ang(_nxt_obs)

