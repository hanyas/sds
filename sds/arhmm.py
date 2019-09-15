import autograd.numpy as np

from sds import HMM
from sds.observations import GaussianObservation
from sds.observations import AutoRegressiveGaussianObservation

from sds.utils import ensure_args_are_viable_lists


class ARHMM(HMM):

    def __init__(self, nb_states, dm_obs, dm_act=0,
                 init_state_prior={}, init_obs_prior={}, trans_prior={}, obs_prior={},
                 init_state_kwargs={}, init_obs_kwargs={}, trans_kwargs={}, obs_kwargs={}):
        super(ARHMM, self).__init__(nb_states, dm_obs, dm_act,
                                    init_state_prior=init_state_prior, trans_prior=trans_prior,
                                    init_state_kwargs=init_state_kwargs, trans_kwargs=trans_kwargs)

        self.init_observation = GaussianObservation(self.nb_states, self.dm_obs, self.dm_act,
                                                    prior=init_obs_prior, **init_obs_kwargs)
        self.observations = AutoRegressiveGaussianObservation(self.nb_states, self.dm_obs, self.dm_act,
                                                              prior=obs_prior, **obs_kwargs)

    @ensure_args_are_viable_lists
    def initialize(self, obs, act=None, **kwargs):
        self.init_observation.initialize(obs, act, kmeans=False)
        self.transitions.initialize(obs, act)
        self.observations.initialize(obs, act)

    @ensure_args_are_viable_lists
    def log_likelihoods(self, obs, act=None):
        loginit = self.init_state.log_init()
        logtrans = self.transitions.log_transition(obs, act)

        ilog = self.init_observation.log_likelihood([_obs[0, :][None, :] for _obs in obs], None)
        arlog = self.observations.log_likelihood(obs, act)

        logobs = []
        for _ilog, _arlog in zip(ilog, arlog):
            logobs.append(np.vstack((_ilog, _arlog)))

        return [loginit, logtrans, logobs]

    @ensure_args_are_viable_lists
    def mean_observation(self, obs, act=None):
        loglikhds = self.log_likelihoods(obs, act)
        alpha = self.forward(loglikhds)
        beta = self.backward(loglikhds)
        gamma = self.marginals(alpha, beta)

        igamma = [_gamma[0, :] for _gamma in gamma]
        iobs = [_obs[0, :] for _obs in obs]
        iact = [_act[0, :] for _act in act]
        imu = self.init_observation.smooth(igamma, iobs, iact)

        armu = self.observations.smooth(gamma, obs, act)

        mu = []
        for _imu, _armu in zip(imu, armu):
            mu.append(np.vstack((_imu, _armu)))

        return mu

    def sample(self, act=None, horizon=None, stoch=True):
        state = []
        obs = []

        for n in range(len(horizon)):
            _act = np.zeros((horizon[n], self.dm_act)) if act is None else act[n]
            _obs = np.zeros((horizon[n], self.dm_obs))
            _state = np.zeros((horizon[n],), np.int64)

            _state[0] = self.init_state.sample()
            _obs[0, :] = self.init_observation.sample(_state[0], x=None, u=None, stoch=stoch)
            for t in range(1, horizon[n]):
                _state[t] = self.transitions.sample(_state[t - 1], _obs[t - 1, :], _act[t - 1, :])
                _obs[t, :] = self.observations.sample(_state[t], _obs[t - 1, :], _act[t - 1, :], stoch=stoch)

            state.append(_state)
            obs.append(_obs)

        return state, obs
