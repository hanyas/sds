import autograd.numpy as np
import autograd.numpy.random as npr

from autograd import value_and_grad
from sds.utils import adam_step, sgd_step

from autograd.scipy.special import logsumexp

from sds.initial import CategoricalInitState
from sds.transitions import StationaryTransition
from sds.observations import GaussianObservation

from sds.utils import ensure_args_are_viable_lists
from sds.cython.hmm_cy import forward_cy, backward_cy

from autograd.tracer import getval
to_c = lambda arr: np.copy(getval(arr), 'C') if not arr.flags['C_CONTIGUOUS'] else getval(arr)


class HMM:

    def __init__(self, nb_states, dm_obs, dm_act=0,
                 init_state_prior={}, trans_prior={}, obs_prior={},
                 init_state_kwargs={}, trans_kwargs={}, obs_kwargs={}):

        self.nb_states = nb_states
        self.dm_obs = dm_obs
        self.dm_act = dm_act

        self.init_state = CategoricalInitState(self.nb_states, prior=init_state_prior, **init_state_kwargs)
        self.transitions = StationaryTransition(self.nb_states, prior=trans_prior, **trans_kwargs)
        self.observations = GaussianObservation(self.nb_states, self.dm_obs, self.dm_act,
                                                prior=obs_prior, **obs_kwargs)

    @property
    def params(self):
        return self.init_state.params, \
               self.transitions.params, \
               self.observations.params

    @params.setter
    def params(self, value):
        self.init_state.params = value[0]
        self.transitions.params = value[1]
        self.observations.params = value[2]

    @ensure_args_are_viable_lists
    def initialize(self, obs, act=None, **kwargs):
        self.init_state.initialize(obs, act)
        self.transitions.initialize(obs, act)
        self.observations.initialize(obs, act)

    def permute(self, perm):
        self.init_state.permute(perm)
        self.transitions.permute(perm)
        self.observations.permute(perm)

    def log_priors(self):
        logprior = 0.0
        logprior += self.init_state.log_prior()
        logprior += self.transitions.log_prior()
        logprior += self.observations.log_prior()
        return logprior

    @ensure_args_are_viable_lists
    def log_likelihoods(self, obs, act=None):
        loginit = self.init_state.log_init()
        logtrans = self.transitions.log_transition(obs, act)
        logobs = self.observations.log_likelihood(obs, act)
        return loginit, logtrans, logobs

    def log_norm(self, obs, act=None):
        loglikhds = self.log_likelihoods(obs, act)
        alpha, norm = self.forward(*loglikhds)
        return sum([np.sum(_norm) for _norm in norm])

    def log_probability(self, obs, act=None):
        return self.log_norm(obs, act) + self.log_priors()

    def forward(self, loginit, logtrans, logobs, logctl=None, cython=True):
        if logctl is None:
            logctl = []
            for _logobs in logobs:
                logctl.append(np.zeros((_logobs.shape[0], self.nb_states)))

        alpha, norm = [], []
        for _logobs, _logctl, _logtrans in zip(logobs, logctl, logtrans):
            T = _logobs.shape[0]
            _alpha = np.zeros((T, self.nb_states))
            _norm = np.zeros((T, ))

            if cython:
                forward_cy(to_c(loginit), to_c(_logtrans),
                           to_c(_logobs), to_c(_logctl),
                           to_c(_alpha), to_c(_norm))
            else:
                for k in range(self.nb_states):
                    _alpha[0, k] = loginit[k] + _logobs[0, k]
                _norm[0] = logsumexp(_alpha[0], axis=-1, keepdims=True)
                _alpha[0] = _alpha[0] - _norm[0]

                _aux = np.zeros((self.nb_states,))
                for t in range(1, T):
                    for k in range(self.nb_states):
                        for j in range(self.nb_states):
                            _aux[j] = _alpha[t - 1, j] + _logtrans[t - 1, j, k]
                        _alpha[t, k] = logsumexp(_aux) + _logobs[t, k] + _logctl[t, k]

                    _norm[t] = logsumexp(_alpha[t], axis=-1, keepdims=True)
                    _alpha[t] = _alpha[t] - _norm[t]

            alpha.append(_alpha)
            norm.append(_norm)
        return alpha, norm

    def backward(self, loginit, logtrans, logobs,
                 logctl=None, scale=None, cython=True):
        if logctl is None:
            logctl = []
            for _logobs in logobs:
                logctl.append(np.zeros((_logobs.shape[0], self.nb_states)))

        beta = []
        for _logobs, _logctl, _logtrans, _scale in zip(logobs, logctl, logtrans, scale):
            T = _logobs.shape[0]
            _beta = np.zeros((T, self.nb_states))

            if cython:
                backward_cy(to_c(loginit), to_c(_logtrans),
                            to_c(_logobs), to_c(_logctl),
                            to_c(_beta), to_c(_scale))
            else:
                for k in range(self.nb_states):
                    _beta[T - 1, k] = 0.0 - _scale[T - 1]

                _aux = np.zeros((self.nb_states,))
                for t in range(T - 2, -1, -1):
                    for k in range(self.nb_states):
                        for j in range(self.nb_states):
                            _aux[j] = _logtrans[t, k, j] + _beta[t + 1, j]\
                                      + _logobs[t + 1, j] + _logctl[t + 1, j]
                        _beta[t, k] = logsumexp(_aux) - _scale[t]

            beta.append(_beta)
        return beta

    def posterior(self, alpha, beta):
        return [np.exp(_alpha + _beta - logsumexp(_alpha + _beta, axis=1,  keepdims=True))
                for _alpha, _beta in zip(alpha, beta)]

    def joint_posterior(self, alpha, beta, loginit, logtrans, logobs, logctl=None):
        if logctl is None:
            logctl = []
            for _logobs in logobs:
                logctl.append(np.zeros((_logobs.shape[0], self.nb_states)))

        zeta = []
        for _logobs, _logctl, _logtrans, _alpha, _beta in\
                zip(logobs, logctl, logtrans, alpha, beta):
            _zeta = _alpha[:-1, :, None] + _beta[1:, None, :] + _logtrans\
                    + _logobs[1:, :][:, None, :] + _logctl[1:, :][:, None, :]

            _zeta -= _zeta.max((1, 2))[:, None, None]
            _zeta = np.exp(_zeta)
            _zeta /= _zeta.sum((1, 2))[:, None, None]

            zeta.append(_zeta)
        return zeta

    def viterbi(self, obs, act=None):
        loginit, logtrans, logobs = self.log_likelihoods(obs, act)[0:3]

        delta = []
        z = []
        for _logobs, _logtrans in zip(logobs, logtrans):
            T = _logobs.shape[0]

            _delta = np.zeros((T, self.nb_states))
            _args = np.zeros((T, self.nb_states), np.int64)
            _z = np.zeros((T, ), np.int64)

            for t in range(T - 2, -1, -1):
                _aux = _logtrans[t, :] + _delta[t + 1, :] + _logobs[t + 1, :]
                _delta[t, :] = np.max(_aux, axis=1)
                _args[t + 1, :] = np.argmax(_aux, axis=1)

            _z[0] = np.argmax(loginit + _delta[0, :] + _logobs[0, :], axis=0)
            for t in range(1, T):
                _z[t] = _args[t, _z[t - 1]]

            delta.append(_delta)
            z.append(_z)

        return delta, z

    def estep(self, obs, act=None):
        loglikhds = self.log_likelihoods(obs, act)
        alpha, norm = self.forward(*loglikhds)
        beta = self.backward(*loglikhds, scale=norm)
        gamma = self.posterior(alpha, beta)
        zeta = self.joint_posterior(alpha, beta, *loglikhds)

        return gamma, zeta

    def mstep(self, gamma, zeta,
              obs, act,
              init_mstep_kwargs,
              trans_mstep_kwargs,
              obs_mstep_kwargs, **kwargs):

        if hasattr(self, 'init_observation'):
            self.init_observation.mstep(gamma, obs, act)

        self.init_state.mstep(gamma, **init_mstep_kwargs)
        self.transitions.mstep(zeta, obs, act, **trans_mstep_kwargs)
        self.observations.mstep(gamma, obs, act, **obs_mstep_kwargs)

    @ensure_args_are_viable_lists
    def em(self, obs, act=None, nb_iter=50, prec=1e-4, verbose=False,
           init_mstep_kwargs={}, trans_mstep_kwargs={}, obs_mstep_kwargs={}, **kwargs):

        lls = []

        ll = self.log_norm(obs, act)
        lls.append(ll)
        if verbose:
            print("it=", 0, "ll=", ll)

        last_ll = ll

        it = 1
        while it <= nb_iter:
            gamma, zeta = self.estep(obs, act)
            self.mstep(gamma, zeta, obs, act,
                       init_mstep_kwargs,
                       trans_mstep_kwargs,
                       obs_mstep_kwargs,
                       **kwargs)

            ll = self.log_norm(obs, act)
            lls.append(ll)
            if verbose:
                print("it=", it, "ll=", ll)

            if (ll - last_ll) < prec:
                break
            else:
                last_ll = ll

            it += 1

        return lls

    @ensure_args_are_viable_lists
    def stochastic_em(self, obs, act=None, nb_epochs=250, batch_size=5,
                      verbose=False, method='adam', **kwargs):

        nb_traj = len(obs)
        nb_points = sum([_obs.shape[0] for _obs in obs])
        nb_batches = int(np.ceil(nb_traj / batch_size))

        def _get_minibatch(itr):
            idx = itr % nb_batches
            sl = slice(idx * batch_size, (idx + 1) * batch_size)
            return obs[sl], act[sl]

        # Define the objective (negative ELBO)
        def _objective(params, itr):
            # Grab a minibatch of data
            obs, act = _get_minibatch(itr)
            nb_steps = sum([_obs.shape[0] for _obs in obs])

            # E step: compute expected latent states with current parameters
            gamma, zeta = self.estep(obs, act)

            # M step: set the parameter and compute the (normalized) objective function
            self.params = params
            loginit, logtrans, logobs = self.log_likelihoods(obs, act)

            # Compute the expected log probability
            # (Scale by number of length of this minibatch.)
            obj = self.log_priors()
            for _gamma, _zeta, _logtrans, _logobs in zip(gamma, zeta, logtrans, logobs):
                obj += np.sum(_gamma[0] * loginit) * nb_traj
                obj += np.sum(_zeta * _logtrans) * (nb_points - nb_traj) / (nb_steps - 1)
                obj += np.sum(_gamma * _logobs) * nb_points / nb_steps
            assert np.isfinite(obj)

            return - obj / nb_points

        lls = [self.log_norm(obs, act)]
        if verbose:
            print("epoch=", 0, "ll=", lls[-1])

        # Run the optimizer
        step = dict(adam=adam_step, sgd=sgd_step)[method]
        state = None
        for itr in range(1, nb_epochs * nb_traj):
            self.params, val, g, state = step(value_and_grad(_objective), self.params,
                                              itr, state=state, **kwargs)
            if itr % nb_traj == 0:
                lls.append(self.log_norm(obs, act))
                if verbose:
                    print("epoch=", itr // nb_traj, "ll=", lls[-1])

        return lls

    @ensure_args_are_viable_lists
    def mean_observation(self, obs, act=None):
        loglikhds = self.log_likelihoods(obs, act)
        alpha, norm = self.forward(*loglikhds)
        beta = self.backward(*loglikhds, scale=norm)
        gamma = self.posterior(alpha, beta)
        return self.observations.smooth(gamma, obs, act)

    @ensure_args_are_viable_lists
    def filter(self, obs, act=None):
        logliklhds = self.log_likelihoods(obs, act)
        alpha, _ = self.forward(*logliklhds)
        belief = [np.exp(_alpha - logsumexp(_alpha, axis=1, keepdims=True)) for _alpha in alpha]
        return belief

    def sample(self, act=None, horizon=None, stoch=True):
        state = []
        obs = []

        for n in range(len(horizon)):
            _act = np.zeros((horizon[n], self.dm_act)) if act is None else act[n]
            _obs = np.zeros((horizon[n], self.dm_obs))
            _state = np.zeros((horizon[n],), np.int64)

            _state[0] = self.init_state.sample()
            _obs[0, :] = self.observations.sample(_state[0], stoch=stoch)
            for t in range(1, horizon[n]):
                _state[t] = self.transitions.sample(_state[t - 1], _obs[t - 1, :], _act[t - 1, :], stoch=stoch)
                _obs[t, :] = self.observations.sample(_state[t], _obs[t - 1, :], _act[t - 1, :], stoch=stoch)

            state.append(_state)
            obs.append(_obs)

        return state, obs

    def step(self, hist_obs=None, hist_act=None, stoch=True, infer='forward'):
        if infer == 'viterbi':
            _, state_seq = self.viterbi(hist_obs, hist_act)
            state = state_seq[0][-1]
        else:
            belief = self.filter(hist_obs, hist_act)
            state = npr.choice(self.nb_states, p=belief[0][-1, ...])

        act = hist_act[-1, :]
        obs = hist_obs[-1, :]

        nxt_state = self.transitions.sample(state, obs, act, stoch=stoch)
        nxt_obs = self.observations.sample(nxt_state, obs, act, stoch=stoch)

        return nxt_state, nxt_obs

    def forcast(self, hist_obs=None, hist_act=None, nxt_act=None,
                horizon=None, stoch=True, infer='forward'):
        nxt_state = []
        nxt_obs = []

        for n in range(len(horizon)):
            _hist_obs = hist_obs[n]
            _hist_act = hist_act[n]

            _nxt_act = np.zeros((horizon[n], self.dm_act)) if nxt_act is None else nxt_act[n]
            _nxt_obs = np.zeros((horizon[n] + 1, self.dm_obs))
            _nxt_state = np.zeros((horizon[n] + 1,), np.int64)

            if infer == 'viterbi':
                _, _state_seq = self.viterbi(_hist_obs, _hist_act)
                _state = _state_seq[0][-1]
            else:
                _belief = self.filter(_hist_obs, _hist_act)
                _state = np.argmax(_belief[0][-1, ...])

            _nxt_state[0] = _state
            _nxt_obs[0, :] = _hist_obs[-1, ...]

            for t in range(horizon[n]):
                _nxt_state[t + 1] = self.transitions.sample(_nxt_state[t], _nxt_obs[t, :], _nxt_act[t, :], stoch=stoch)
                _nxt_obs[t + 1, :] = self.observations.sample(_nxt_state[t + 1], _nxt_obs[t, :], _nxt_act[t, :], stoch=stoch)

            nxt_state.append(_nxt_state)
            nxt_obs.append(_nxt_obs)

        return nxt_state, nxt_obs

    @ensure_args_are_viable_lists
    def kstep_mse(self, obs, act, horizon=1, stoch=True, infer='forward'):
        from sklearn.metrics import mean_squared_error, explained_variance_score

        mse, norm_mse = [], []
        for _obs, _act in zip(obs, act):
            _hist_obs, _hist_act, _nxt_act = [], [], []
            _target, _prediction = [], []

            _nb_steps = _obs.shape[0] - horizon
            for t in range(_nb_steps):
                _hist_obs.append(_obs[:t + 1, :])
                _hist_act.append(_act[:t + 1, :])
                _nxt_act.append(_act[t: t + horizon, :])

            _hr = [horizon for _ in range(_nb_steps)]
            _, _obs_hat = self.forcast(hist_obs=_hist_obs, hist_act=_hist_act,
                                       nxt_act=_nxt_act, horizon=_hr,
                                       stoch=stoch, infer=infer)

            for t in range(_nb_steps):
                _target.append(_obs[t + horizon, :])
                _prediction.append(_obs_hat[t][-1, :])

            _target = np.vstack(_target)
            _prediction = np.vstack(_prediction)

            _mse = mean_squared_error(_target, _prediction)
            _norm_mse = explained_variance_score(_target, _prediction, multioutput='variance_weighted')

            mse.append(_mse)
            norm_mse.append(_norm_mse)

        return np.mean(mse), np.mean(norm_mse)
