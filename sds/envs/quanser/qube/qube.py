import numpy as np

from sds.envs.quanser.common import LabeledBox
from sds.envs.quanser.qube.base import QubeBase


def normalize(x):
    return (x % (2 * np.pi)) - np.pi


class Qube(QubeBase):
    def __init__(self, fs, fs_ctrl):
        super(Qube, self).__init__(fs, fs_ctrl)
        self._sim_state = None

        self.dm_obs = 4
        self.dm_act = 1

        self._dt = 0.01

        obs_max = np.array([2.3, np.inf, 30., 40.])
        self.observation_space = LabeledBox(
            labels=('theta', 'alpha', 'th_d', 'al_d'),
            low=-obs_max, high=obs_max, dtype=np.float64)

    @property
    def ulim(self):
        return self.action_space.high

    def _observation(self, state):
        obs = np.float64([state[0], normalize(state[1]),
                          state[2], state[3]])
        return obs


class QubeWithCartesianObservation(QubeBase):
    def __init__(self, fs, fs_ctrl):
        super(QubeWithCartesianObservation, self).__init__(fs, fs_ctrl)
        self._sim_state = None

        self.dm_obs = 5
        self.dm_act = 1

        self._dt = 0.01

        obs_max = np.array([2.3, 1., 1., 30., 40.])
        self.observation_space = LabeledBox(
            labels=('theta', 'cos_al', 'sin_al', 'th_d', 'al_d'),
            low=-obs_max, high=obs_max, dtype=np.float64)

    @property
    def ulim(self):
        return self.action_space.high

    def _observation(self, state):
        obs = np.float64([state[0],
                          np.cos(state[1]),
                          np.sin(state[1]),
                          state[2],
                          state[3]])
        return obs
