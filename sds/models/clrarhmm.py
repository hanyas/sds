import numpy as np
import numpy.random as npr

from scipy.special import logsumexp

from sds.models import RecurrentAutoRegressiveHiddenMarkovModel

from sds.initial import InitGaussianControl
from sds.initial import BayesianInitGaussianControl
from sds.initial import BayesianInitGaussianControlWithAutomaticRelevance
from sds.controls import AutorRegressiveLinearGaussianControl
from sds.controls import BayesianAutorRegressiveLinearGaussianControl
from sds.controls import BayesianAutoRegressiveLinearGaussianControlWithAutomaticRelevance

from sds.controls import LinearGaussianControl
from sds.controls import BayesianLinearGaussianControl
from sds.controls import BayesianLinearGaussianControlWithAutomaticRelevance

from sds.utils.decorate import ensure_args_are_viable
from sds.cython.clhmm_cy import forward_cy, backward_cy

from tqdm import trange

to_c = lambda arr: np.copy(arr, 'C') \
    if not arr.flags['C_CONTIGUOUS'] else arr


class ClosedLoopRecurrentAutoRegressiveHiddenMarkovModel:

    def __init__(self, nb_states, obs_dim, act_dim, obs_lag=1,
                 algo_type='MAP', init_obs_type='full',
                 trans_type='neural', obs_type='full', ctl_type='full',
                 init_state_prior=None, init_obs_prior=None,
                 trans_prior=None, obs_prior=None, ctl_prior=None,
                 init_state_kwargs={}, init_obs_kwargs={},
                 trans_kwargs={}, obs_kwargs={}, ctl_kwargs={}):

        self.nb_states = nb_states
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.obs_lag = obs_lag

        self.algo_type = algo_type

        self.dynamics = RecurrentAutoRegressiveHiddenMarkovModel(nb_states, obs_dim, act_dim, obs_lag,
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
            if self.ctl_type == 'full':
                self.controls = BayesianLinearGaussianControl(self.nb_states, self.obs_dim, self.act_dim,
                                                              prior=ctl_prior, **ctl_kwargs)
            elif self.ctl_type == 'ard':
                self.controls = BayesianLinearGaussianControlWithAutomaticRelevance(self.nb_states, self.obs_dim, self.act_dim,
                                                                                    prior=ctl_prior, **ctl_kwargs)

    @property
    def params(self):
        return self.dynamics.params,\
               self.controls.params

    @params.setter
    def params(self, value):
        self.dynamics.params = value[:4]
        self.controls.params = value[4]

    def permute(self, perm):
        self.dynamics.permute(perm)
        self.controls.permute(perm)

    @ensure_args_are_viable
    def initialize(self, obs, act, **kwargs):
        self.dynamics.initialize(obs, act, **kwargs)
        self.controls.initialize(obs, act, **kwargs)

    @ensure_args_are_viable
    def log_likelihoods(self, obs, act):
        if isinstance(obs, np.ndarray) and isinstance(act, np.ndarray):
            loginit, logtrans, logobs = self.dynamics.log_likelihoods(obs, act)
            logact = self.controls.log_likelihood(obs, act)
            return loginit, logtrans, logobs, logact
        else:
            def inner(obs, act):
                return self.log_likelihoods.__wrapped__(self, obs, act)
            result = map(inner, obs, act)
            return list(map(list, zip(*result)))

    def log_normalizer(self, obs, act):
        loglikhds = self.log_likelihoods(obs, act)
        _, norm = self.forward(*loglikhds)
        return np.sum(np.hstack(norm))

    def forward(self, loginit, logtrans, logobs, logact):
        if isinstance(loginit, np.ndarray) \
                and isinstance(logtrans, np.ndarray) \
                and isinstance(logobs, np.ndarray) \
                and isinstance(logact, np.ndarray):

            nb_steps = logobs.shape[0]
            alpha = np.zeros((nb_steps, self.nb_states))
            norm = np.zeros((nb_steps,))

            forward_cy(to_c(loginit), to_c(logtrans),
                       to_c(logobs), to_c(logact),
                       to_c(alpha), to_c(norm))

            return alpha, norm
        else:
            def partial(loginit, logtrans, logobs, logact):
                return self.forward(loginit, logtrans, logobs, logact)
            result = map(partial, loginit, logtrans, logobs, logact)
            return list(map(list, zip(*result)))

    def backward(self, loginit, logtrans, logobs, logact, scale=None):
        if isinstance(loginit, np.ndarray) \
                and isinstance(logtrans, np.ndarray) \
                and isinstance(logobs, np.ndarray) \
                and isinstance(logact, np.ndarray) \
                and isinstance(scale, np.ndarray):

            nb_steps = logobs.shape[0]
            beta = np.zeros((nb_steps, self.nb_states))

            backward_cy(to_c(loginit), to_c(logtrans),
                        to_c(logobs), to_c(logact),
                        to_c(beta), to_c(scale))

            return beta
        else:
            def partial(loginit, logtrans, logobs, logact, scale):
                return self.backward(loginit, logtrans, logobs, logact, scale)
            return list(map(partial, loginit, logtrans, logobs, logact, scale))

    def posterior(self, alpha, beta, temperature=1.):
        if isinstance(alpha, np.ndarray) and isinstance(beta, np.ndarray):
            return np.exp(temperature * (alpha + beta)
                          - logsumexp(temperature * (alpha + beta), axis=1, keepdims=True))
        else:
            def partial(alpha, beta):
                return self.posterior(alpha, beta, temperature)
            return list(map(self.posterior, alpha, beta))

    def joint_posterior(self, alpha, beta, loginit, logtrans, logobs, logact, temperature=1.):
        if isinstance(loginit, np.ndarray) \
                and isinstance(logtrans, np.ndarray) \
                and isinstance(logobs, np.ndarray) \
                and isinstance(logact, np.ndarray) \
                and isinstance(alpha, np.ndarray) \
                and isinstance(beta, np.ndarray):

            zeta = temperature * (alpha[:-1, :, None] + beta[1:, None, :]) + logtrans \
                   + logobs[1:][:, None, :] + logact[1:][:, None, :]

            return np.exp(zeta - logsumexp(zeta, axis=(1, 2), keepdims=True))
        else:
            def partial(alpha, beta, loginit, logtrans, logobs, logact):
                return self.joint_posterior(alpha, beta, loginit, logtrans, logobs, logact, temperature)
            return list(map(partial, alpha, beta, loginit, logtrans, logobs, logact))

    def estep(self, obs, act, temperature=1.):
        loglikhds = self.log_likelihoods(obs, act)
        alpha, norm = self.forward(*loglikhds)
        beta = self.backward(*loglikhds, scale=norm)
        gamma = self.posterior(alpha, beta, temperature=temperature)
        zeta = self.joint_posterior(alpha, beta, *loglikhds,
                                    temperature=temperature)
        return gamma, zeta

    def mstep(self, gamma, zeta, obs, act,
              init_state_mstep_kwargs,
              init_obs_mstep_kwargs,
              trans_mstep_kwargs,
              obs_mstep_kwargs,
              ctl_mstep_kwargs, **kwargs):

            self.dynamics.mstep(gamma, zeta, obs, act,
                                init_state_mstep_kwargs,
                                trans_mstep_kwargs,
                                obs_mstep_kwargs,
                                init_obs_mstep_kwargs)

            self.controls.mstep(gamma, obs, act, **ctl_mstep_kwargs)

    @ensure_args_are_viable
    def em(self, train_obs, train_act=None,
           nb_iter=50, prec=1e-4, initialize=True,
           init_state_mstep_kwargs={},
           init_obs_mstep_kwargs={},
           trans_mstep_kwargs={},
           obs_mstep_kwargs={},
           ctl_mstep_kwargs={}, **kwargs):

        proc_id = kwargs.pop('proc_id', 0)

        if initialize:
            self.initialize(train_obs, train_act)

        train_lls = []
        train_ll = self.log_normalizer(train_obs, train_act)
        train_lls.append(train_ll)
        last_train_ll = train_ll

        pbar = trange(nb_iter, position=proc_id)
        pbar.set_description("#{}, ll: {:.5f}".format(proc_id, train_lls[-1]))

        for _ in pbar:
            gamma, zeta = self.estep(train_obs, train_act)
            self.mstep(gamma, zeta,
                       train_obs, train_act,
                       init_state_mstep_kwargs,
                       init_obs_mstep_kwargs,
                       trans_mstep_kwargs,
                       obs_mstep_kwargs,
                       ctl_mstep_kwargs)

            train_ll = self.log_normalizer(train_obs, train_act)
            train_lls.append(train_ll)

            pbar.set_description("#{}, ll: {:.5f}".format(proc_id, train_lls[-1]))

            if abs(train_ll - last_train_ll) < prec:
                break
            else:
                last_train_ll = train_ll

        return train_lls

    @ensure_args_are_viable
    def smoothed_control(self, obs, act):
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
    def filtered_control(self, obs, act, stoch=False):
        if isinstance(obs, np.ndarray) and isinstance(act, np.ndarray):
            loglikhds = self.dynamics.log_likelihoods(obs, act)
            alpha, _ = self.dynamics.forward(*loglikhds)

            w = np.exp(alpha - logsumexp(alpha, axis=-1, keepdims=True))
            z = np.zeros((len(obs,)), dtype=np.int64)
            u = np.zeros((len(act), self.act_dim))

            for t in range(len(obs)):
                z[t] = npr.choice(self.nb_states, p=w[t, :]) if stoch\
                       else np.argmax(w[t, :])
                u[t] = self.controls.sample(z[t], obs[t, :]) if stoch\
                       else self.controls.mean(z[t], obs[t, :])
            return z, u
        else:
            def partial(obs, act):
                return self.filtered_control.__wrapped__(self, obs, act, stoch)
            result = map(partial, obs, act)
            return list(map(list, zip(*result)))

    def action(self, hist_obs, hist_act, stoch=False, average=False):
        obs = hist_obs[-1]
        belief = self.dynamics.filtered_state(hist_obs, hist_act)[-1]
        state = npr.choice(self.nb_states, p=belief) if stoch else np.argmax(belief)

        nxt_act = np.zeros((self.act_dim,))
        if average:
            for k in range(self.nb_states):
                nxt_act += belief[k] * self.controls.sample(k, obs) if stoch\
                           else self.controls.mean(k, obs)
        else:
            nxt_act = self.controls.sample(state, obs) if stoch\
                      else self.controls.mean(state, obs)

        return belief, state, nxt_act


class AutoRegressiveClosedLoopRecurrentHiddenMarkovModel(ClosedLoopRecurrentAutoRegressiveHiddenMarkovModel):

    def __init__(self, nb_states, obs_dim, act_dim, obs_lag=1, ctl_lag=1,
                 algo_type='MAP', init_obs_type='full', init_ctl_type='full',
                 trans_type='neural', obs_type='full', ctl_type='full',
                 init_state_prior=None, init_obs_prior=None, init_ctl_prior=None,
                 trans_prior=None, obs_prior=None, ctl_prior=None,
                 init_state_kwargs={}, init_obs_kwargs={}, init_ctl_kwargs={},
                 trans_kwargs={}, obs_kwargs={}, ctl_kwargs={}):

        super(AutoRegressiveClosedLoopRecurrentHiddenMarkovModel, self).__init__(nb_states, obs_dim, act_dim, obs_lag, algo_type,
                                                                                 init_obs_type=init_obs_type, trans_type=trans_type,
                                                                                 obs_type=obs_type, ctl_type=None,
                                                                                 init_state_prior=init_state_prior, init_obs_prior=init_obs_prior,
                                                                                 trans_prior=trans_prior, obs_prior=obs_prior, ctl_prior=None,
                                                                                 init_state_kwargs=init_state_kwargs, init_obs_kwargs=init_obs_kwargs,
                                                                                 trans_kwargs=trans_kwargs, obs_kwargs=obs_kwargs, ctl_kwargs={})

        self.init_ctl_type = init_ctl_type
        self.init_ctl_prior = init_ctl_prior
        self.init_ctl_kwargs = init_ctl_kwargs

        self.ctl_type = ctl_type
        self.ctl_prior = ctl_prior
        self.ctl_kwargs = ctl_kwargs

        self.ctl_lag = ctl_lag

        if self.algo_type == 'ML':
            self.init_control = InitGaussianControl(self.nb_states, self.obs_dim, self.act_dim, self.ctl_lag, **init_ctl_kwargs)
            self.controls = AutorRegressiveLinearGaussianControl(self.nb_states, self.obs_dim, self.act_dim, nb_lags=self.ctl_lag, **ctl_kwargs)
        else:
            if self.init_ctl_type == 'full':
                self.init_control = BayesianInitGaussianControl(self.nb_states, self.obs_dim, self.act_dim,
                                                                self.ctl_lag, prior=init_ctl_prior, **init_ctl_kwargs)
            elif self.init_ctl_type == 'ard':
                self.init_control = BayesianInitGaussianControlWithAutomaticRelevance(self.nb_states, self.obs_dim, self.act_dim,
                                                                                      self.ctl_lag, prior=init_ctl_prior, **init_ctl_kwargs)

            if self.ctl_type == 'full':
                self.controls = BayesianAutorRegressiveLinearGaussianControl(self.nb_states, self.obs_dim, self.act_dim,
                                                                             self.ctl_lag, prior=ctl_prior,  **ctl_kwargs)
            elif self.ctl_type == 'ard':
                self.controls = BayesianAutoRegressiveLinearGaussianControlWithAutomaticRelevance(self.nb_states, self.obs_dim, self.act_dim,
                                                                                                  self.ctl_lag, prior=ctl_prior,  **ctl_kwargs)

    @property
    def params(self):
        return self.dynamics.init_state.params, \
               self.dynamics.init_observation.params,\
               self.init_control.params,\
               self.dynamics.transitions.params, \
               self.dynamics.observations.params,\
               self.controls.params

    @params.setter
    def params(self, value):
        self.dynamics.init_state.params = value[0]
        self.dynamics.init_observation.params = value[1]
        self.init_control.params = value[2]
        self.dynamics.transitions.params = value[3]
        self.dynamics.observations.params = value[4]
        self.controls.params = value[5]

    def permute(self, perm):
        self.dynamics.permute(perm)
        self.init_control.permute(perm)
        self.controls.permute(perm)

    @ensure_args_are_viable
    def initialize(self, obs, act=None, **kwargs):
        self.dynamics.initialize(obs, act, **kwargs)
        self.init_control.initialize(obs, act, **kwargs)
        self.controls.initialize(obs, act, **kwargs)

    @ensure_args_are_viable
    def log_likelihoods(self, obs, act=None):
        if isinstance(obs, np.ndarray) and isinstance(act, np.ndarray):
            loginit, logtrans, logobs = self.dynamics.log_likelihoods(obs, act)
            ilogact = self.init_control.log_likelihood(obs, act)
            arlogact = self.controls.log_likelihood(obs, act)
            logact = np.vstack((ilogact, arlogact))
            return loginit, logtrans, logobs, logact
        else:
            def inner(obs, act):
                return self.log_likelihoods.__wrapped__(self, obs, act)
            result = map(inner, obs, act)
            return list(map(list, zip(*result)))

    def mstep(self, gamma, zeta, obs, act,
              init_state_mstep_kwargs,
              init_obs_mstep_kwargs,
              trans_mstep_kwargs,
              obs_mstep_kwargs,
              ctl_mstep_kwargs,
              init_ctl_mstep_kwargs={}):

        self.dynamics.mstep(gamma, zeta, obs, act,
                            init_state_mstep_kwargs,
                            trans_mstep_kwargs,
                            obs_mstep_kwargs,
                            init_obs_mstep_kwargs)

        self.init_control.mstep(gamma, obs, act, **init_ctl_mstep_kwargs)
        self.controls.mstep(gamma, obs, act, **ctl_mstep_kwargs)

    @ensure_args_are_viable
    def smoothed_control(self, obs, act=None):
        if isinstance(obs, np.ndarray) and isinstance(act, np.ndarray):
            loglikhds = self.log_likelihoods(obs, act)
            alpha, norm = self.forward(*loglikhds)
            beta = self.backward(*loglikhds, scale=norm)
            gamma = self.posterior(alpha, beta)

            iact = self.init_control.smooth(gamma, obs, act)
            aract = self.controls.smooth(gamma, obs, act)
            return np.vstack((iact, aract))
        else:
            def inner(obs, act):
                return self.smoothed_control.__wrapped__(self, obs, act)
            return list(map(inner, obs, act))

    @ensure_args_are_viable
    def filtered_control(self, obs, act=None, stoch=False):
        if isinstance(obs, np.ndarray) and isinstance(act, np.ndarray):
            loglikhds = self.dynamics.log_likelihoods(obs, act)
            alpha, _ = self.dynamics.forward(*loglikhds)

            w = np.exp(alpha - logsumexp(alpha, axis=-1, keepdims=True))
            z = np.zeros((len(obs,)), dtype=np.int64)
            u = np.zeros((len(act), self.act_dim))

            for t in range(self.ctl_lag):
                z[t] = npr.choice(self.nb_states, p=w[t, :]) if stoch\
                       else np.argmax(w[t, :])
                u[t] = self.init_control.sample(z[t], obs[t, :]) if stoch\
                       else self.init_control.mean(z[t], obs[t, :])

            for t in range(self.ctl_lag, len(obs)):
                z[t] = npr.choice(self.nb_states, p=w[t, :]) if stoch\
                       else np.argmax(w[t, :])
                u[t] = self.controls.sample(z[t], obs[t - self.ctl_lag:t + 1], ar=True) if stoch\
                       else self.controls.mean(z[t], obs[t - self.ctl_lag:t + 1], ar=True)
            return z, u
        else:
            def partial(obs, act):
                return self.filtered_control.__wrapped__(self, obs, act, stoch)
            result = map(partial, obs, act)
            return list(map(list, zip(*result)))

    def action(self, hist_obs, hist_act, stoch=False, average=False):
        belief = self.dynamics.filtered_state(hist_obs, hist_act)[-1]
        state = npr.choice(self.nb_states, p=belief) if stoch else np.argmax(belief)

        if len(hist_obs) <= self.ctl_lag:
            obs = hist_obs[-1]
            nxt_act = np.zeros((self.act_dim,))
            if average:
                for k in range(self.nb_states):
                    nxt_act += belief[k] * self.init_control.sample(k, obs) if stoch\
                               else self.init_control.mean(k, obs)
            else:
                nxt_act = self.init_control.sample(state, obs) if stoch\
                          else self.init_control.mean(state, obs)
        else:
            obs = hist_obs[-1 - self.ctl_lag:]
            nxt_act = np.zeros((self.act_dim,))
            if average:
                for k in range(self.nb_states):
                    nxt_act += belief[k] * self.controls.sample(k, obs, ar=True) if stoch\
                               else self.controls.mean(k, obs, ar=True)
            else:
                nxt_act = self.controls.sample(state, obs, ar=True) if stoch\
                          else self.controls.mean(state, obs, ar=True)

        return belief, state, nxt_act


class HybridController:

    def __init__(self, dynamics, ctl_type='full', ctl_prior=None, ctl_kwargs={}):

        self.dynamics = dynamics

        self.nb_states = dynamics.nb_states
        self.obs_dim = dynamics.obs_dim
        self.act_dim = dynamics.act_dim
        self.obs_lag = dynamics.obs_lag

        self.algo_type = dynamics.algo_type

        self.ctl_type = ctl_type
        self.ctl_prior = ctl_prior
        self.ctl_kwargs = ctl_kwargs

        if self.algo_type == 'ML':
            self.controls = LinearGaussianControl(self.nb_states, self.obs_dim, self.act_dim, **ctl_kwargs)
        else:
            if self.ctl_type == 'full':
                self.controls = BayesianLinearGaussianControl(self.nb_states, self.obs_dim, self.act_dim,
                                                              prior=ctl_prior, **ctl_kwargs)
            elif self.ctl_type == 'ard':
                self.controls = BayesianLinearGaussianControlWithAutomaticRelevance(self.nb_states, self.obs_dim, self.act_dim,
                                                                                    prior=ctl_prior, **ctl_kwargs)

    @property
    def params(self):
        return self.controls.params

    @params.setter
    def params(self, value):
        self.controls.params = value

    def permute(self, perm):
        pass

    @ensure_args_are_viable
    def initialize(self, obs, act, **kwargs):
        self.controls.initialize(obs, act, **kwargs)

    @ensure_args_are_viable
    def log_likelihoods(self, obs, act):
        if isinstance(obs, np.ndarray) and isinstance(act, np.ndarray):
            loginit, logtrans, logobs = self.dynamics.log_likelihoods(obs, act)
            logact = self.controls.log_likelihood(obs, act)
            return loginit, logtrans, logobs, logact
        else:
            def inner(obs, act):
                return self.log_likelihoods.__wrapped__(self, obs, act)
            result = map(inner, obs, act)
            return list(map(list, zip(*result)))

    def log_normalizer(self, obs, act):
        loglikhds = self.log_likelihoods(obs, act)
        _, norm = self.forward(*loglikhds)
        return np.sum(np.hstack(norm))


    def forward(self, loginit, logtrans, logobs, logact):
        if isinstance(loginit, np.ndarray) \
                and isinstance(logtrans, np.ndarray) \
                and isinstance(logobs, np.ndarray) \
                and isinstance(logact, np.ndarray):

            nb_steps = logobs.shape[0]
            alpha = np.zeros((nb_steps, self.nb_states))
            norm = np.zeros((nb_steps,))

            forward_cy(to_c(loginit), to_c(logtrans),
                       to_c(logobs), to_c(logact),
                       to_c(alpha), to_c(norm))

            return alpha, norm
        else:
            def partial(loginit, logtrans, logobs, logact):
                return self.forward(loginit, logtrans, logobs, logact)
            result = map(partial, loginit, logtrans, logobs, logact)
            return list(map(list, zip(*result)))

    def backward(self, loginit, logtrans, logobs, logact, scale=None):
        if isinstance(loginit, np.ndarray) \
                and isinstance(logtrans, np.ndarray) \
                and isinstance(logobs, np.ndarray) \
                and isinstance(logact, np.ndarray) \
                and isinstance(scale, np.ndarray):

            nb_steps = logobs.shape[0]
            beta = np.zeros((nb_steps, self.nb_states))

            backward_cy(to_c(loginit), to_c(logtrans),
                        to_c(logobs), to_c(logact),
                        to_c(beta), to_c(scale))

            return beta
        else:
            def partial(loginit, logtrans, logobs, logact, scale):
                return self.backward(loginit, logtrans, logobs, logact, scale)
            return list(map(partial, loginit, logtrans, logobs, logact, scale))

    def posterior(self, alpha, beta, temperature=1.):
        if isinstance(alpha, np.ndarray) and isinstance(beta, np.ndarray):
            return np.exp(temperature * (alpha + beta)
                          - logsumexp(temperature * (alpha + beta), axis=1, keepdims=True))
        else:
            def partial(alpha, beta):
                return self.posterior(alpha, beta, temperature)
            return list(map(self.posterior, alpha, beta))

    def joint_posterior(self, alpha, beta, loginit, logtrans, logobs, logact, temperature=1.):
        if isinstance(loginit, np.ndarray) \
                and isinstance(logtrans, np.ndarray) \
                and isinstance(logobs, np.ndarray) \
                and isinstance(logact, np.ndarray) \
                and isinstance(alpha, np.ndarray) \
                and isinstance(beta, np.ndarray):

            zeta = temperature * (alpha[:-1, :, None] + beta[1:, None, :]) + logtrans \
                   + logobs[1:][:, None, :] + logact[1:][:, None, :]

            return np.exp(zeta - logsumexp(zeta, axis=(1, 2), keepdims=True))
        else:
            def partial(alpha, beta, loginit, logtrans, logobs, logact):
                return self.joint_posterior(alpha, beta, loginit, logtrans, logobs, logact, temperature)
            return list(map(partial, alpha, beta, loginit, logtrans, logobs, logact))

    def estep(self, obs, act, temperature=1.):
        loglikhds = self.log_likelihoods(obs, act)
        alpha, norm = self.forward(*loglikhds)
        beta = self.backward(*loglikhds, scale=norm)
        gamma = self.posterior(alpha, beta, temperature=temperature)
        zeta = self.joint_posterior(alpha, beta, *loglikhds,
                                    temperature=temperature)
        return gamma, zeta

    def mstep(self, gamma, obs, act,
              ctl_mstep_kwargs, **kwargs):

            self.controls.mstep(gamma, obs, act, **ctl_mstep_kwargs)

    def weighted_mstep(self, gamma, obs, act, weights,
                       ctl_mstep_kwargs, **kwargs):

            self.controls.weighted_mstep(gamma, obs, act, weights, **ctl_mstep_kwargs)

    @ensure_args_are_viable
    def em(self, train_obs, train_act,
           nb_iter=50, prec=1e-4, initialize=False,
           ctl_mstep_kwargs={}, **kwargs):

        proc_id = kwargs.pop('proc_id', 0)

        if initialize:
            self.initialize(train_obs, train_act)

        train_lls = []
        train_ll = self.log_normalizer(train_obs, train_act)
        train_lls.append(train_ll)
        last_train_ll = train_ll

        pbar = trange(nb_iter, position=proc_id)
        pbar.set_description("#{}, ll: {:.5f}".format(proc_id, train_lls[-1]))

        for _ in pbar:
            gamma, zeta = self.estep(train_obs, train_act)
            self.mstep(gamma, train_obs, train_act,
                       ctl_mstep_kwargs)

            train_ll = self.log_normalizer(train_obs, train_act)
            train_lls.append(train_ll)

            pbar.set_description("#{}, ll: {:.5f}".format(proc_id, train_lls[-1]))

            if abs(train_ll - last_train_ll) < prec:
                break
            else:
                last_train_ll = train_ll

        return train_lls

    def action(self, hist_obs, hist_act, stoch=False, average=False):
        obs = hist_obs[-1]
        belief = self.dynamics.filtered_state(hist_obs, hist_act)[-1]
        state = npr.choice(self.nb_states, p=belief) if stoch else np.argmax(belief)

        nxt_act = np.zeros((self.act_dim,))
        if average:
            for k in range(self.nb_states):
                nxt_act += belief[k] * self.controls.sample(k, obs) if stoch \
                    else self.controls.mean(k, obs)
        else:
            nxt_act = self.controls.sample(state, obs) if stoch \
                else self.controls.mean(state, obs)

        return belief, state, nxt_act
