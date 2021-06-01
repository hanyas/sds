import numpy as np
import numpy.random as npr

from scipy.special import logsumexp

from sds.models import RecurrentAutoRegressiveHiddenMarkovModel

# from sds.initial import InitGaussianControl
# from sds.initial import BayesianInitGaussianControl
# from sds.controls import AutorRegressiveLinearGaussianControl

from sds.controls import LinearGaussianControl, BayesianLinearGaussianControl

from sds.utils.decorate import ensure_args_are_viable


class ClosedLoopRecurrentAutoRegressiveHiddenMarkovModel(RecurrentAutoRegressiveHiddenMarkovModel):

    def __init__(self, nb_states, obs_dim, act_dim, obs_lag=1,
                 algo_type='MAP', init_obs_type='full',
                 trans_type='neural', obs_type='full', ctl_type='full',
                 init_state_prior=None, init_obs_prior=None, init_ctl_prior=None,
                 trans_prior=None, obs_prior=None, ctl_prior=None,
                 init_state_kwargs={}, init_obs_kwargs={},
                 trans_kwargs={}, obs_kwargs={}, ctl_kwargs={},
                 infer_dyn=True, infer_ctl=True):

        super(ClosedLoopRecurrentAutoRegressiveHiddenMarkovModel, self).__init__(nb_states, obs_dim, act_dim, obs_lag,
                                                                                 algo_type=algo_type, init_obs_type=init_obs_type,
                                                                                 trans_type=trans_type, obs_type=obs_type,
                                                                                 init_state_prior=init_state_prior,
                                                                                 init_obs_prior=init_obs_prior,
                                                                                 trans_prior=trans_prior,
                                                                                 obs_prior=obs_prior,
                                                                                 init_state_kwargs=init_state_kwargs,
                                                                                 init_obs_kwargs=init_obs_kwargs,
                                                                                 trans_kwargs=trans_kwargs,
                                                                                 obs_kwargs=obs_kwargs)

        self.ctl_type = ctl_type
        self.ctl_prior = ctl_prior
        self.ctl_kwargs = ctl_kwargs

        if self.algo_type == 'ML':
            self.controls = LinearGaussianControl(self.nb_states, self.obs_dim, self.act_dim, **ctl_kwargs)
        else:
            self.controls = BayesianLinearGaussianControl(self.nb_states, self.obs_dim, self.act_dim,
                                                          prior=ctl_prior, **ctl_kwargs)

        self.infer_dyn = infer_dyn
        self.infer_ctl = infer_ctl

    @property
    def dynamics(self):
        rarhmm = RecurrentAutoRegressiveHiddenMarkovModel(self.nb_states, self.obs_dim, self.act_dim, self.obs_lag,
                                                          algo_type=self.algo_type, init_obs_type=self.init_obs_type,
                                                          trans_type=self.trans_type, obs_type=self.obs_type,
                                                          init_state_prior=self.init_state_prior,
                                                          init_obs_prior=self.init_obs_prior,
                                                          trans_prior=self.trans_prior,
                                                          obs_prior=self.obs_prior,
                                                          init_state_kwargs=self.init_state_kwargs,
                                                          init_obs_kwargs=self.init_obs_kwargs,
                                                          trans_kwargs=self.trans_kwargs,
                                                          obs_kwargs=self.obs_kwargs)

        rarhmm.init_state = self.init_state
        rarhmm.init_observation = self.init_observation
        rarhmm.transitions = self.transitions
        rarhmm.observations = self.observations
        return rarhmm

    @dynamics.setter
    def dynamics(self, model):
        assert isinstance(model, RecurrentAutoRegressiveHiddenMarkovModel)
        assert model.nb_states == self.nb_states
        assert model.obs_dim == self.obs_dim
        assert model.act_dim == self.act_dim
        assert model.obs_lag == self.obs_lag
        assert model.algo_type == self.algo_type

        self.init_state = model.init_state
        self.init_observation = model.init_observation
        self.transitions = model.transitions
        self.observations = model.observations

    @property
    def params(self):
        return super(ClosedLoopRecurrentAutoRegressiveHiddenMarkovModel, self).params,\
               self.controls.params

    @params.setter
    def params(self, value):
        super(ClosedLoopRecurrentAutoRegressiveHiddenMarkovModel, self).params = value[:4]
        self.controls.params = value[4]

    def permute(self, perm):
        super(ClosedLoopRecurrentAutoRegressiveHiddenMarkovModel, self).permute(perm)
        self.controls.permute(perm)

    @ensure_args_are_viable
    def initialize(self, obs, act=None, **kwargs):
        super(ClosedLoopRecurrentAutoRegressiveHiddenMarkovModel, self).initialize(obs, act, **kwargs)
        if self.infer_ctl:
            self.controls.initialize(obs, act, **kwargs)

    @ensure_args_are_viable
    def log_likelihoods(self, obs, act=None):
        if isinstance(obs, np.ndarray) and isinstance(act, np.ndarray):
            loginit, logtrans, logobs = super(ClosedLoopRecurrentAutoRegressiveHiddenMarkovModel, self).log_likelihoods(obs, act)
            if self.infer_ctl:
                logctl = self.controls.log_likelihood(obs, act)
                return loginit, logtrans, logobs, logctl
            else:
                return loginit, logtrans, logobs
        else:
            def inner(obs, act):
                return self.log_likelihoods.__wrapped__(self, obs, act)
            result = map(inner, obs, act)
            return list(map(list, zip(*result)))

    def mstep(self, gamma, zeta,
              obs, act,
              init_state_mstep_kwargs,
              trans_mstep_kwargs,
              obs_mstep_kwargs, **kwargs):

        if self.infer_dyn:
            super(RecurrentAutoRegressiveHiddenMarkovModel, self).mstep(gamma, zeta, obs, act,
                                                                        init_state_mstep_kwargs,
                                                                        trans_mstep_kwargs,
                                                                        obs_mstep_kwargs, **kwargs)
        if self.infer_ctl:
            ctl_mstep_kwargs = kwargs.get('ctl_mstep_kwargs', {})
            self.controls.mstep(gamma, obs, act, **ctl_mstep_kwargs)

    @ensure_args_are_viable
    def smoothed_control(self, obs, act=None):
        if isinstance(obs, np.ndarray) and isinstance(act, np.ndarray):
            loglikhds = self.log_likelihoods(obs, act)
            alpha, norm = self.forward(*loglikhds)
            beta = self.backward(*loglikhds, scale=norm)
            gamma = self.posterior(alpha, beta)
            return self.controls.smooth(gamma, obs, act)
        else:
            def inner(obs, act):
                return self.smoothed_control.__wrapped__(self, obs, act)
            return list(map(inner, obs, act))

    @ensure_args_are_viable
    def filtered_control(self, obs, act=None, stoch=False):
        if isinstance(obs, np.ndarray) and isinstance(act, np.ndarray):
            loglikhds = self.log_likelihoods(obs, act)
            alpha, _ = self.forward(*loglikhds)

            w = np.exp(alpha - logsumexp(alpha, axis=-1, keepdims=True))
            z = np.zeros((len(obs,)), dtype=np.int64)
            u = np.zeros((len(act), self.act_dim))

            for t in range(len(obs)):
                z[t] = npr.choice(self.nb_states, p=w[t, :]) if stoch\
                    else np.argmax(w[t, :])
                u[t, :] = self.controls.sample(z[t], obs[t, :]) if stoch\
                          else self.controls.mean(z[t], obs[t, :])
            return z, u
        else:
            def partial(obs, act):
                return self.filtered_control.__wrapped__(self, obs, act, stoch)
            result = map(partial, obs, act)
            return list(map(list, zip(*result)))


# class AutoRegressiveClosedLoopRecurrentHiddenMarkovModel(RecurrentAutoRegressiveHiddenMarkovModel):


# class ClosedLoopRecurrentAutoRegressiveHiddenMarkovModel(RecurrentAutoRegressiveHiddenMarkovModel):
#
#     def __init__(self, nb_states, obs_dim, act_dim, obs_lag=1, ctl_lag=0,
#                  algo_type='MAP', init_obs_type='full', init_ctl_type='full',
#                  trans_type='neural', obs_type='full', ctl_type='full',
#                  init_state_prior={}, init_obs_prior={}, init_ctl_prior={},
#                  trans_prior={}, obs_prior={}, ctl_prior={},
#                  init_state_kwargs={}, init_obs_kwargs={}, init_ctl_kwargs={},
#                  trans_kwargs={}, obs_kwargs={}, ctl_kwargs={},
#                  infer_dyn=True, infer_ctl=False):
#
#         super(ClosedLoopRecurrentAutoRegressiveHiddenMarkovModel, self).__init__(nb_states, obs_dim, act_dim, obs_lag,
#                                                                                  algo_type=algo_type, init_obs_type=init_obs_type,
#                                                                                  trans_type=trans_type, obs_type=obs_type,
#                                                                                  init_state_prior=init_state_prior,
#                                                                                  init_obs_prior=init_obs_prior,
#                                                                                  trans_prior=trans_prior,
#                                                                                  obs_prior=obs_prior,
#                                                                                  init_state_kwargs=init_state_kwargs,
#                                                                                  init_obs_kwargs=init_obs_kwargs,
#                                                                                  trans_kwargs=trans_kwargs,
#                                                                                  obs_kwargs=obs_kwargs)
#
#         self.init_ctl_type = init_ctl_type
#         self.ctl_type = ctl_type
#
#         self.ctl_lag = ctl_lag
#         self.arctl = True if self.ctl_lag > 0 else False
#
#         if self.arctl:
#             if self.algo_type == 'ML':
#                 self.init_control = InitGaussianControl(self.nb_states, self.obs_dim, self.act_dim, self.ctl_lag, **init_ctl_kwargs)
#                 self.controls = AutorRegressiveLinearGaussianControl(self.nb_states, self.obs_dim, self.act_dim, nb_lags=self.ctl_lag, **ctl_kwargs)
#             else:
#                 self.init_control = BayesianInitGaussianControl(self.nb_states, self.obs_dim, self.act_dim,
#                                                                 self.ctl_lag, prior=init_ctl_prior, **init_ctl_kwargs)
#                 self.controls = AutorRegressiveLinearGaussianControl(self.nb_states, self.obs_dim, self.act_dim, nb_lags=self.ctl_lag, **ctl_kwargs)
#         else:
#             self.controls = LinearGaussianControl(self.nb_states, self.obs_dim, self.act_dim, **ctl_kwargs)
#
#         self.infer_dyn = infer_dyn
#         self.infer_ctl = infer_ctl
#
#     @property
#     def params(self):
#         return self.init_state.params, \
#                self.init_observation.params,\
#                self.transitions.params, \
#                self.observations.params
#
#     @params.setter
#     def params(self, value):
#         self.init_state.params = value[0]
#         self.init_observation.params = value[1]
#         self.transitions.params = value[2]
#         self.observations.params = value[3]
#
#     @ensure_args_are_viable
#     def initialize(self, obs, act=None, **kwargs):
#         super(ClosedLoopRecurrentAutoRegressiveHiddenMarkovModel, self).initialize(obs, act, **kwargs)
#         if self.infer_ctl:
#             if self.arctl:
#                 self.init_control.initialize(obs, act, **kwargs)
#             self.controls.initialize(obs, act, **kwargs)
#
#     @ensure_args_are_viable
#     def log_likelihoods(self, obs, act=None):
#         loginit, logtrans, logobs = super(ClosedLoopRecurrentAutoRegressiveHiddenMarkovModel, self).log_likelihoods(obs, act)
#         if self.infer_ctl:
#             if self.arctl:
#                 ilog = self.init_control.log_likelihood(obs, act)
#                 arlog = self.controls.log_likelihood(obs, act)
#                 logctl = [np.vstack((i, ar)) for i, ar in zip(ilog, arlog)]
#             else:
#                 logctl = self.controls.log_likelihood(obs, act)
#             return loginit, logtrans, logobs, logctl
#         else:
#             return loginit, logtrans, logobs
#
#     def mstep(self, gamma, zeta,
#               obs, act,
#               init_mstep_kwargs,
#               trans_mstep_kwargs,
#               obs_mstep_kwargs, **kwargs):
#
#         if self.infer_dyn:
#             self.init_state.mstep(gamma, **init_mstep_kwargs)
#             self.init_observation.mstep(gamma, obs)
#             self.transitions.mstep(zeta, obs, act, **trans_mstep_kwargs)
#             self.observations.mstep(gamma, obs, act, **obs_mstep_kwargs)
#         if self.infer_ctl:
#             if self.arctl:
#                 init_ctl_mstep_kwargs = kwargs.get('init_ctl_mstep_kwargs', {})
#                 self.init_control.mstep(gamma, obs, act, **init_ctl_mstep_kwargs)
#             ctl_mstep_kwargs = kwargs.get('ctl_mstep_kwargs', {})
#             self.controls.mstep(gamma, obs, act, **ctl_mstep_kwargs)
#
#     def permute(self, perm):
#         super(ClosedLoopRecurrentAutoRegressiveHiddenMarkovModel, self).permute(perm)
#         self.controls.permute(perm)
#
#     @ensure_args_are_viable
#     def mean_control(self, obs, act=None):
#         loglikhds = self.log_likelihoods(obs, act)
#         alpha, norm = self.forward(*loglikhds)
#         beta = self.backward(*loglikhds, scale=norm)
#         gamma = self.posterior(alpha, beta)
#         if self.arctl:
#             imu = self.init_control.smooth(gamma, obs)
#             armu = self.controls.smooth(gamma, obs, act)
#             mean = [np.vstack((i, ar)) for i, ar in zip(imu, armu)]
#         else:
#             mean = self.controls.smooth(gamma, obs, act)
#         return mean
#
#     @ensure_args_are_viable
#     def filter_control(self, obs, act=None, stoch=False):
#         loglikhds = self.log_likelihoods(obs, act)
#         alpha, _ = self.forward(*loglikhds)
#
#         state, ctl = [], []
#         for _alpha, _obs, _act in zip(alpha, obs, act):
#             w = np.exp(_alpha - logsumexp(_alpha, axis=-1, keepdims=True))
#             z = np.zeros((len(_obs,)), dtype=np.int64)
#             u = np.zeros((len(_act), self.act_dim))
#
#             for t in range(len(_obs)):
#                 z[t] = npr.choice(self.nb_states, p=w[t, :]) if stoch\
#                     else np.argmax(w[t, :])
#                 if self.arctl:
#                     if t < self.ctl_lag:
#                         u[t, :] = self.init_control.sample(z[t], _obs[t, :]) if stoch\
#                                   else self.init_control.mean(z[t], _obs[t, :])
#                     else:
#                         u[t, :] = self.controls.sample(z[t], _obs[t - self.ctl_lag:t + 1]) if stoch\
#                                   else self.controls.mean(z[t], _obs[t - self.ctl_lag:t + 1])
#                 else:
#                     u[t, :] = self.controls.sample(z[t], _obs[t, :]) if stoch\
#                               else self.controls.mean(z[t], _obs[t, :])
#
#             state.append(z)
#             ctl.append(u)
#         return state, ctl
