import numpy as np
import numpy.random as npr

from scipy.special import logsumexp

from sds.initial import InitCategoricalState
from sds.transitions import StationaryTransition
from sds.observations import GaussianObservation

from sds.utils.decorate import ensure_args_are_viable
from sds.utils.general import find_permutation
from sds.cython.hmm_cy import forward_cy, backward_cy
from sds.cython.hmm_cy import backward_sample_cy

from tqdm import trange

to_c = lambda arr: np.copy(arr, 'C') \
    if not arr.flags['C_CONTIGUOUS'] else arr


class HiddenMarkovModel:

    def __init__(self, nb_states, obs_dim, act_dim=0,
                 init_state_kwargs={}, trans_kwargs={}, obs_kwargs={}):

        self.nb_states = nb_states
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.init_state = InitCategoricalState(self.nb_states, **init_state_kwargs)
        self.transitions = StationaryTransition(self.nb_states, self.obs_dim,
                                                self.act_dim, **trans_kwargs)
        self.observations = GaussianObservation(self.nb_states, self.obs_dim,
                                                self.act_dim, **obs_kwargs)

    @property
    def params(self):
        return self.init_state.params,\
               self.transitions.params,\
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

    @ensure_args_are_viable
    def initialize(self, obs, act=None, **kwargs):
        self.init_state.initialize()
        self.transitions.initialize(obs, act)
        self.observations.initialize(obs, act)

    @ensure_args_are_viable
    def log_likelihoods(self, obs, act=None):
        if isinstance(obs, np.ndarray) and isinstance(act, np.ndarray):
            loginit = self.init_state.log_init()
            logtrans = self.transitions.log_transition(obs, act)
            logobs = self.observations.log_likelihood(obs, act)
            return loginit, logtrans, logobs
        else:
            def inner(obs, act):
                return self.log_likelihoods.__wrapped__(self, obs, act)
            result = map(inner, obs, act)
            return list(map(list, zip(*result)))

    def log_normalizer(self, obs, act=None):
        loglikhds = self.log_likelihoods(obs, act)
        _, norm = self.forward(*loglikhds)
        return np.sum(np.hstack(norm))

    def forward(self, loginit, logtrans, logobs, cython=True):
        if isinstance(loginit, np.ndarray)\
                and isinstance(logtrans, np.ndarray)\
                and isinstance(logobs, np.ndarray):

            nb_steps = logobs.shape[0]
            alpha = np.zeros((nb_steps, self.nb_states))
            norm = np.zeros((nb_steps, ))

            if cython:
                forward_cy(to_c(loginit), to_c(logtrans),
                           to_c(logobs), to_c(alpha), to_c(norm))
            else:
                for k in range(self.nb_states):
                    alpha[0, k] = loginit[k] + logobs[0, k]

                norm[0] = logsumexp(alpha[0], axis=-1, keepdims=True)
                alpha[0] = alpha[0] - norm[0]

                aux = np.zeros((self.nb_states,))
                for t in range(1, nb_steps):
                    for k in range(self.nb_states):
                        for j in range(self.nb_states):
                            aux[j] = alpha[t - 1, j] + logtrans[t - 1, j, k]
                        alpha[t, k] = logsumexp(aux) + logobs[t, k]

                    norm[t] = logsumexp(alpha[t], axis=-1, keepdims=True)
                    alpha[t] = alpha[t] - norm[t]

            return alpha, norm
        else:
            def partial(loginit, logtrans, logobs):
                return self.forward(loginit, logtrans, logobs, cython)
            result = map(partial, loginit, logtrans, logobs)
            return list(map(list, zip(*result)))

    def backward(self, loginit, logtrans, logobs, scale=None, cython=True):
        if isinstance(loginit, np.ndarray)\
                and isinstance(logtrans, np.ndarray)\
                and isinstance(logobs, np.ndarray)\
                and isinstance(scale, np.ndarray):

            nb_steps = logobs.shape[0]
            beta = np.zeros((nb_steps, self.nb_states))

            if cython:
                backward_cy(to_c(loginit), to_c(logtrans),
                            to_c(logobs), to_c(beta), to_c(scale))
            else:
                for k in range(self.nb_states):
                    beta[nb_steps - 1, k] = 0.0 - scale[nb_steps - 1]

                aux = np.zeros((self.nb_states, ))
                for t in range(nb_steps - 2, -1, -1):
                    for k in range(self.nb_states):
                        for j in range(self.nb_states):
                            aux[j] = logtrans[t, k, j] + beta[t + 1, j] \
                                     + logobs[t + 1, j]
                        beta[t, k] = logsumexp(aux) - scale[t]

            return beta
        else:
            def partial(loginit, logtrans, logobs, scale):
                return self.backward(loginit, logtrans, logobs, scale, cython)
            return list(map(partial, loginit, logtrans, logobs, scale))

    def smoothed_posterior(self, alpha, beta, temperature=1.):
        if isinstance(alpha, np.ndarray) and isinstance(beta, np.ndarray):
            return np.exp(temperature * (alpha + beta)
                          - logsumexp(temperature * (alpha + beta), axis=1, keepdims=True))
        else:
            def partial(alpha, beta):
                return self.smoothed_posterior(alpha, beta, temperature)
            return list(map(self.smoothed_posterior, alpha, beta))

    def smoothed_joint(self, alpha, beta, loginit, logtrans, logobs, temperature=1.):
        if isinstance(loginit, np.ndarray)\
                and isinstance(logtrans, np.ndarray)\
                and isinstance(logobs, np.ndarray)\
                and isinstance(alpha, np.ndarray)\
                and isinstance(beta, np.ndarray):

            zeta = temperature * (alpha[:-1, :, None] + beta[1:, None, :])\
                   + logtrans + logobs[1:][:, None, :]

            return np.exp(zeta - logsumexp(zeta, axis=(1, 2), keepdims=True))
        else:
            def partial(alpha, beta, loginit, logtrans, logobs):
                return self.smoothed_joint(alpha, beta, loginit, logtrans, logobs, temperature)
            return list(map(partial, alpha, beta, loginit, logtrans, logobs))

    @ensure_args_are_viable
    def sampled_posterior(self, obs, act=None, cython=True):
        if isinstance(obs, np.ndarray) and isinstance(act, np.ndarray):
            loglikhds = self.log_likelihoods(obs, act)
            alpha, norm = self.forward(*loglikhds)

            loginit, logtrans, logobs = loglikhds

            nb_steps = logobs.shape[0]
            states = -1 * np.ones((nb_steps, ), np.int64)

            if cython:
                backward_sample_cy(to_c(logtrans), to_c(logobs),
                                   to_c(alpha), to_c(states))
            else:
                T = logobs.shape[0]
                K = logobs.shape[1]

                states[-1] = np.random.choice(K, size=1, p=np.exp(alpha[-1]))

                logdist = np.zeros(K)
                for t in range(T - 2, -1, -1):
                    j = states[t + 1]
                    for k in range(K):
                        logdist[k] = logobs[t + 1, j] + logtrans[t, k, j] \
                                     + alpha[t, k] - alpha[t + 1, j]

                    logdist -= logsumexp(logdist)
                    states[t] = np.random.choice(K, size=1, p=np.exp(logdist))

            return states
        else:
            def partial(obs, act):
                return self.sampled_posterior.__wrapped__(self, obs, act, cython)
            return list(map(partial, obs, act))

    @ensure_args_are_viable
    def viterbi(self, obs, act=None):
        if isinstance(obs, np.ndarray) and isinstance(act, np.ndarray):

            loginit, logtrans, logobs = self.log_likelihoods(obs, act)

            nb_steps = logobs.shape[0]

            delta = np.zeros((nb_steps, self.nb_states))
            args = np.zeros((nb_steps, self.nb_states), np.int64)
            z = -1 * np.ones((nb_steps, ), np.int64)

            for t in range(nb_steps - 2, -1, -1):
                aux = logobs[t + 1] + logtrans[t] + delta[t + 1]
                delta[t] = np.max(aux, axis=1)
                args[t + 1] = np.argmax(aux, axis=1)

            z[0] = np.argmax(logobs[0] + loginit + delta[0], axis=0)
            for t in range(1, nb_steps):
                z[t] = args[t, z[t - 1]]

            return delta, z
        else:
            def inner(obs, act):
                return self.viterbi.__wrapped__(self, obs, act)
            result = map(inner, obs, act)
            return list(map(list, zip(*result)))

    def estep(self, obs, act=None, temperature=1.):
        loglikhds = self.log_likelihoods(obs, act)
        alpha, norm = self.forward(*loglikhds)
        beta = self.backward(*loglikhds, scale=norm)
        gamma = self.smoothed_posterior(alpha, beta, temperature=temperature)
        zeta = self.smoothed_joint(alpha, beta, *loglikhds, temperature=temperature)
        return gamma, zeta

    def mstep(self, gamma, zeta,
              obs, act,
              init_state_mstep_kwargs,
              trans_mstep_kwargs,
              obs_mstep_kwargs, **kwargs):

        self.init_state.mstep(gamma, **init_state_mstep_kwargs)
        self.transitions.mstep(zeta, obs, act, **trans_mstep_kwargs)
        self.observations.mstep(gamma, obs, act, **obs_mstep_kwargs)

    @ensure_args_are_viable
    def em(self, train_obs, train_act=None,
           nb_iter=50, prec=1e-4, initialize=True,
           init_state_mstep_kwargs={},
           trans_mstep_kwargs={},
           obs_mstep_kwargs={}, **kwargs):

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

    @ensure_args_are_viable
    def annealed_em(self, train_obs, train_act=None,
                    nb_iter=50, nb_sub_iter=50,
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

    @ensure_args_are_viable
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

    @ensure_args_are_viable
    def smoothed_observation(self, obs, act=None):
        if isinstance(obs, np.ndarray) and isinstance(act, np.ndarray):
            loglikhds = self.log_likelihoods(obs, act)
            alpha, norm = self.forward(*loglikhds)
            beta = self.backward(*loglikhds, scale=norm)
            gamma = self.smoothed_posterior(alpha, beta)
            return self.observations.smooth(gamma, obs, act)
        else:
            def inner(obs, act):
                return self.smoothed_observation.__wrapped__(self, obs, act)
            return list(map(inner, obs, act))

    @ensure_args_are_viable
    def filtered_posterior(self, obs, act=None):
        if isinstance(obs, np.ndarray) and isinstance(act, np.ndarray):
            # pad action for filtering
            if len(obs) != len(act):
                act = np.pad(act, [(0, 1), (0, 0)], 'constant')

            logliklhds = self.log_likelihoods(obs, act)
            alpha, _ = self.forward(*logliklhds)
            return np.exp(alpha - logsumexp(alpha, axis=1, keepdims=True))
        else:
            def inner(obs, act):
                return self.filtered_posterior.__wrapped__(self, obs, act)
            return list(map(inner, obs, act))

    def sample(self, horizon, act=None, seed=None):
        if isinstance(horizon, int):
            assert isinstance(act, np.ndarray) or act is None

            npr.seed(seed)

            act = np.zeros((horizon, self.act_dim)) if act is None else act
            obs = np.zeros((horizon, self.obs_dim))
            state = np.zeros((horizon,), np.int64)

            state[0] = self.init_state.sample()
            obs[0] = self.observations.sample(state[0])
            for t in range(1, horizon):
                state[t] = self.transitions.sample(state[t - 1], obs[t - 1], act[t - 1])
                obs[t] = self.observations.sample(state[t], obs[t - 1], act[t - 1])

            return state, obs
        else:
            seeds = [i for i in range(len(horizon))]
            act = [None] * len(horizon) if act is None else act
            res = list(map(self.sample, horizon, act, seeds))
            return list(map(list, zip(*res)))

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
            self.permute(find_permutation(true_state, state,
                                          K1=self.nb_states,
                                          K2=self.nb_states))
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
            mean = self.smoothed_observation(obs, act)
            for k in range(self.obs_dim):
                axes[k].plot(mean[:, k], '-k', lw=1)

        if self.act_dim > 0:
            for k in range(self.act_dim):
                axes[self.obs_dim + k].plot(act[:, k], '-r', lw=2)

        k = self.obs_dim + self.act_dim
        axes[k].imshow(state[None], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors) - 1)
        axes[k].set_xlim(0, len(obs))
        axes[k].set_ylabel("$z_{\\mathrm{inf}}$")
        axes[k].set_yticks([])

        if true_state is not None:
            k = self.obs_dim + self.act_dim + 1
            axes[k].imshow(true_state[None], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors) - 1)
            axes[k].set_xlim(0, len(obs))
            axes[k].set_ylabel("$z_{\\mathrm{true}}$")
            axes[k].set_yticks([])

        axes[-1].set_xlabel("time")

        plt.tight_layout()
        plt.show()
