import numpy as np
import numpy.random as npr

from scipy.special import logsumexp

from sds.initial import InitCategoricalState
from sds.transitions import StationaryTransition
from sds.observations import GaussianObservation

from sds.utils.decorate import ensure_args_are_viable_lists
from sds.utils.decorate import init_empty_logctl_to_zero
from sds.utils.general import find_permutation
from sds.cython.hmm_cy import forward_cy, backward_cy

from tqdm import trange
from pathos.multiprocessing import ProcessPool

to_c = lambda arr: np.copy(arr, 'C') \
    if not arr.flags['C_CONTIGUOUS'] else arr


class HiddenMarkovModel:

    def __init__(self, nb_states, obs_dim, act_dim=0,
                 init_state_kwargs={}, trans_kwargs={}, obs_kwargs={}):

        self.nb_states = nb_states
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.init_state = InitCategoricalState(self.nb_states, **init_state_kwargs)
        self.transitions = StationaryTransition(self.nb_states, **trans_kwargs)
        self.observations = GaussianObservation(self.nb_states, self.obs_dim,
                                                self.act_dim, **obs_kwargs)

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

    def permute(self, perm):
        self.init_state.permute(perm)
        self.transitions.permute(perm)
        self.observations.permute(perm)

    @ensure_args_are_viable_lists
    def initialize(self, obs, act=None, **kwargs):
        self.init_state.initialize()
        self.transitions.initialize(obs, act)
        self.observations.initialize(obs, act)

    @ensure_args_are_viable_lists
    def log_likelihoods(self, obs, act=None):
        loginit = self.init_state.log_init()
        logtrans = self.transitions.log_transition(obs, act)
        logobs = self.observations.log_likelihood(obs, act)
        return loginit, logtrans, logobs

    def log_normalizer(self, obs, act=None):
        loglikhds = self.log_likelihoods(obs, act)
        _, norm = self.forward(*loglikhds)
        return np.sum(np.hstack(norm))

    @init_empty_logctl_to_zero
    def forward(self, loginit, logtrans, logobs,
                logctl=None, cython=True):

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

    @init_empty_logctl_to_zero
    def backward(self, loginit, logtrans, logobs,
                 logctl=None, scale=None, cython=True):

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
                            _aux[j] = _logtrans[t, k, j] + _beta[t + 1, j] \
                                      + _logobs[t + 1, j] + _logctl[t + 1, j]
                        _beta[t, k] = logsumexp(_aux) - _scale[t]

            beta.append(_beta)
        return beta

    @staticmethod
    def posterior(alpha, beta, temperature=1.):
        return [np.exp(temperature * (_alpha + _beta) - logsumexp(temperature * (_alpha + _beta), axis=1, keepdims=True))
                for _alpha, _beta in zip(alpha, beta)]

    @init_empty_logctl_to_zero
    def joint_posterior(self, alpha, beta, loginit, logtrans,
                        logobs, logctl=None, temperature=1.):

        zeta = []
        for _logobs, _logctl, _logtrans, _alpha, _beta in \
                zip(logobs, logctl, logtrans, alpha, beta):
            _zeta = temperature * (_alpha[:-1, :, None] + _beta[1:, None, :]) + _logtrans \
                    + _logobs[1:, :][:, None, :] + _logctl[1:, :][:, None, :]

            zeta.append(np.exp(_zeta - logsumexp(_zeta, axis=(1, 2), keepdims=True)))

        return zeta

    @ensure_args_are_viable_lists
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

    def estep(self, obs, act=None, temperature=1.):
        loglikhds = self.log_likelihoods(obs, act)
        alpha, norm = self.forward(*loglikhds)
        beta = self.backward(*loglikhds, scale=norm)
        gamma = self.posterior(alpha, beta, temperature=temperature)
        zeta = self.joint_posterior(alpha, beta, *loglikhds, temperature=temperature)
        return gamma, zeta

    def mstep(self, gamma, zeta,
              obs, act,
              init_state_mstep_kwargs,
              trans_mstep_kwargs,
              obs_mstep_kwargs, **kwargs):

        self.init_state.mstep(gamma, **init_state_mstep_kwargs)
        self.transitions.mstep(zeta, obs, act, **trans_mstep_kwargs)
        self.observations.mstep(gamma, obs, act, **obs_mstep_kwargs)

    @ensure_args_are_viable_lists
    def em(self, train_obs, train_act=None,
           nb_iter=50, prec=1e-4, initialize=True,
           init_state_mstep_kwargs={},
           trans_mstep_kwargs={},
           obs_mstep_kwargs={}, **kwargs):

        proc_id = kwargs.get('proc_id', 0)

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
                       trans_mstep_kwargs,
                       obs_mstep_kwargs,
                       **kwargs)

            train_ll = self.log_normalizer(train_obs, train_act)
            train_lls.append(train_ll)

            pbar.set_description("#{}, ll: {:.5f}".format(proc_id, train_lls[-1]))

            if abs(train_ll - last_train_ll) < prec:
                break
            else:
                last_train_ll = train_ll

        return train_lls

    @ensure_args_are_viable_lists
    def annealed_em(self, train_obs, train_act=None,
                    nb_iter=50, nb_sub_iter=25,
                    prec=1e-4, discount=0.99,
                    init_state_mstep_kwargs={},
                    trans_mstep_kwargs={},
                    obs_mstep_kwargs={}, **kwargs):

        proc_id = kwargs.get('proc_id', 0)

        train_lls = []
        train_ll = self.log_normalizer(train_obs, train_act)
        train_lls.append(train_ll)
        last_train_ll = train_ll

        pbar = trange(nb_iter, position=proc_id)
        pbar.set_description("#{}, ll: {:.5f}".format(proc_id, train_lls[-1]))

        for i in pbar:
            temperature = 1. - np.power(discount, i)
            for j in range(nb_sub_iter):
                gamma, zeta = self.estep(train_obs, train_act, temperature)
                self.mstep(gamma, zeta,
                           train_obs, train_act,
                           init_state_mstep_kwargs,
                           trans_mstep_kwargs,
                           obs_mstep_kwargs,
                           **kwargs)

            train_ll = self.log_normalizer(train_obs, train_act)
            train_lls.append(train_ll)

            pbar.set_description("#{}, ll: {:.5f}, tmp: {:.3f}".format(proc_id, train_lls[-1], temperature))

            if abs(train_ll - last_train_ll) < prec:
                break
            else:
                last_train_ll = train_ll

        return train_lls

    @ensure_args_are_viable_lists
    def earlystop_em(self, train_obs, train_act=None,
                     nb_iter=50, prec=1e-4, initialize=True,
                     init_state_mstep_kwargs={}, trans_mstep_kwargs={},
                     obs_mstep_kwargs={}, test_obs=None, test_act=None,
                     **kwargs):

        assert test_obs is not None and test_act is not None

        proc_id = kwargs.get('proc_id', 0)

        if initialize:
            self.initialize(train_obs, train_act)

        nb_train = np.vstack(train_obs).shape[0]
        nb_test = np.vstack(test_obs).shape[0]
        nb_all = nb_train + nb_test

        train_lls = []
        train_ll = self.log_normalizer(train_obs, train_act)
        train_lls.append(train_ll)
        last_train_ll = train_ll

        test_lls = []
        test_ll = self.log_normalizer(test_obs, test_act)
        test_lls.append(test_ll)
        last_test_ll = test_ll

        all_ll = last_train_ll + last_test_ll

        score = (all_ll - train_ll) / (nb_all - nb_train)
        last_score = score

        pbar = trange(nb_iter, position=proc_id)
        pbar.set_description("#{}, train_ll: {:.5f}, test_ll: {:.5f}, "
                             "score: {:.5f}".format(proc_id, train_ll, test_ll, score))

        for _ in pbar:
            gamma, zeta = self.estep(train_obs, train_act)
            self.mstep(gamma, zeta, train_obs, train_act,
                       init_state_mstep_kwargs,
                       trans_mstep_kwargs,
                       obs_mstep_kwargs,
                       **kwargs)

            train_ll = self.log_normalizer(train_obs, train_act)
            train_lls.append(train_ll)

            test_ll = self.log_normalizer(test_obs, test_act)
            test_lls.append(test_ll)

            all_ll = train_ll + test_ll
            score = (all_ll - train_ll) / (nb_all - nb_train)

            pbar.set_description("#{}, train_ll: {:.5f}, test_ll: {:.5f}, "
                                 "score: {:.5f}".format(proc_id, train_ll, test_ll, score))

            if abs(score - last_score) < prec:
                break
            else:
                last_score = score

        return train_lls

    @ensure_args_are_viable_lists
    def mean_observation(self, obs, act=None):
        loglikhds = self.log_likelihoods(obs, act)
        alpha, norm = self.forward(*loglikhds)
        beta = self.backward(*loglikhds, scale=norm)
        gamma = self.posterior(alpha, beta)
        mean_obs = self.observations.smooth(gamma, obs, act)
        return mean_obs

    @ensure_args_are_viable_lists
    def filter(self, obs, act=None):
        logliklhds = self.log_likelihoods(obs, act)
        alpha, _ = self.forward(*logliklhds)
        belief = [np.exp(_alpha - logsumexp(_alpha, axis=1, keepdims=True))
                  for _alpha in alpha]
        return belief

    def _sample(self, horizon, act=None, seed=None):
        npr.seed(seed)

        act = np.zeros((horizon, self.act_dim)) if act is None else act
        obs = np.zeros((horizon, self.obs_dim))
        state = np.zeros((horizon,), np.int64)

        state[0] = self.init_state.sample()
        obs[0, :] = self.observations.sample(state[0])
        for t in range(1, horizon):
            state[t] = self.transitions.sample(state[t - 1], obs[t - 1, :], act[t - 1, :])
            obs[t, :] = self.observations.sample(state[t], obs[t - 1, :], act[t - 1, :])

        return state, obs

    def sample(self, horizon, act=None, nodes=8):
        act = [None] * len(horizon) if act is None else act
        seeds = [i for i in range(len(horizon))]

        pool = ProcessPool(nodes=nodes)
        res = pool.map(self._sample, horizon, act, seeds)
        pool.clear()

        state, obs = list(map(list, zip(*res)))
        return state, obs

    def plot(self, obs, act=None, true_state=None, plot_mean=True, title=None):

        import seaborn as sns
        import matplotlib.pyplot as plt
        import matplotlib.colors as cls

        sns.set_style("white")
        sns.set_context("talk")

        color_names = ["windows blue", "red", "amber", "faded green", "dusty purple",
                       "greyish", "lightblue",  "magenta", "clay", "teal",
                       "marine blue", "orangered", "burnt yellow",  "jungle green"]

        colors = sns.xkcd_palette(color_names)
        cmap = cls.ListedColormap(colors)

        _, state = self.viterbi(obs, act)
        if true_state is not None:
            self.permute(find_permutation(true_state, state[0],
                                          K1=self.nb_states, K2=self.nb_states))
            _, state = self.viterbi(obs, act)

        nb_plots = self.obs_dim + self.act_dim + 1
        if true_state is not None:
            nb_plots += 1

        fig, axes = plt.subplots(nrows=nb_plots, ncols=1, figsize=(8, 8))
        if title is not None:
            fig.suptitle(title)
        for k in range(self.obs_dim):
            axes[k].plot(obs[:, k], '-b', lw=2)

        if plot_mean:
            mean = self.mean_observation(obs, act)
            for k in range(self.obs_dim):
                axes[k].plot(mean[0][:, k], '-k', lw=1)

        if self.act_dim > 0:
            for k in range(self.act_dim):
                axes[self.obs_dim + k].plot(act[:, k], '-r', lw=2)

        k = self.obs_dim + self.act_dim
        axes[k].imshow(state[0][None, :], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors) - 1)
        axes[k].set_xlim(0, len(obs))
        axes[k].set_ylabel("$z_{\\mathrm{inf}}$")
        axes[k].set_yticks([])

        if true_state is not None:
            k = self.obs_dim + self.act_dim + 1
            axes[k].imshow(true_state[None, :], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors) - 1)
            axes[k].set_xlim(0, len(obs))
            axes[k].set_ylabel("$z_{\\mathrm{true}}$")
            axes[k].set_yticks([])

        axes[-1].set_xlabel("time")

        plt.tight_layout()
        plt.show()
