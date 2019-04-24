import autograd.numpy as np
import autograd.numpy.random as npr

from scipy.special import logsumexp

from sds.distn import CategoricalInitState
from sds.distn import RecurrentTransition, RecurrentOnlyTransition
from sds.distn import NeuralRecurrentTransition, NeuralRecurrentOnlyTransition
from sds.distn import GaussianObservation, AutoRegressiveGaussianObservation

from sds.util import normalize, find_permutation
from sds.util import fit_linear_regression


class rARHMM:

    def __init__(self, nb_states, dim_obs):
        self.nb_states = nb_states
        self.dim_obs = dim_obs

        # init state
        self.init_state = CategoricalInitState(self.nb_states)

        # transitions
        # self.transitions = RecurrentTransition(self.nb_states, self.dim_obs, degree=1)
        self.transitions = RecurrentOnlyTransition(self.nb_states, self.dim_obs, degree=1)

        # self.transitions = NeuralRecurrentTransition(self.nb_states, self.dim_obs, hidden_layer_sizes=(10, ))
        # self.transitions = NeuralRecurrentOnlyTransition(self.nb_states, self.dim_obs, hidden_layer_sizes=(5, ))

        # init observation
        self.init_observation = GaussianObservation(nb_states=1, dim_obs=self.dim_obs)

        # observations
        self.observations = AutoRegressiveGaussianObservation(self.nb_states, self.dim_obs)

    def sample(self, T):
        obs = np.empty((T, self.dim_obs))
        states = np.empty((T, ), np.int64)

        states[0] = self.init_state.sample()
        obs[0, :] = self.init_observation.sample(z=0)
        for t in range(1, T):
            states[t] = self.transitions.sample(states[t - 1], obs[t - 1: t + 1, :])
            obs[t, :] = self.observations.sample(states[t], obs[t - 1, :])

        return states, obs

    def initialize(self, obs):
        T = obs.shape[0]

        self.init_observation.mu = npr.randn(1, self.dim_obs)
        self.init_observation.cov = np.array([np.eye(self.dim_obs, self.dim_obs)])

        aux = np.zeros((self.nb_states, self.dim_obs, self.dim_obs))
        for k in range(self.nb_states):
            idx = npr.choice(T - 1, replace=False, size=(T - 1)//self.nb_states)
            x, y = obs[idx, :], obs[idx + 1, :]

            coef_, intercept_, sigmas = fit_linear_regression(x, y)
            self.observations.A[k, ...] = coef_
            self.observations.c[k, :] = intercept_
            aux[k, ...] = np.diag(sigmas)

        self.observations.cov = aux

    def logpriors(self):
        logprior = 0.0
        logprior += self.init_state.logprior()
        logprior += self.transitions.logprior()
        logprior += self.observations.logprior()
        return logprior

    def loglikhds(self, obs):
        loginit = self.init_state.loglik()
        logtrans = self.transitions.loglik(obs)

        ilog = np.array([self.init_observation.loglik(obs[0, :])
                         for _ in range(self.nb_states)]).T
        arlog = self.observations.loglik(obs)
        logobs = np.concatenate((ilog, arlog))

        return [loginit, logtrans, logobs]

    def forward(self, loglikhds):
        loginit, logtrans, logobs = loglikhds
        T = logobs.shape[0]

        alpha = np.zeros((T, self.nb_states))

        for k in range(self.nb_states):
            alpha[0, k] = loginit[k] + logobs[0, k]

        aux = np.zeros((self.nb_states,))
        for t in range(1, T):
            for k in range(self.nb_states):
                for j in range(self.nb_states):
                    aux[j] = alpha[t - 1, j] + logtrans[t - 1, j, k]
                alpha[t, k] = logsumexp(aux) + logobs[t, k]

        return alpha

    def backward(self, loglikhds):
        loginit, logtrans, logobs = loglikhds
        T = logobs.shape[0]

        beta = np.zeros((T, self.nb_states))
        for k in range(self.nb_states):
            beta[T - 1, k] = 0.0

        aux = np.zeros((self.nb_states,))
        for t in range(T - 2, -1, -1):
            for k in range(self.nb_states):
                for j in range(self.nb_states):
                    aux[j] = logtrans[t, k, j] + beta[t + 1, j] + logobs[t + 1, j]
                beta[t, k] = logsumexp(aux)

        return beta

    def expected(self, alpha, beta):
        return np.exp(alpha + beta - logsumexp(alpha + beta, axis=1,  keepdims=True))

    def joint(self, loglikhds, alpha, beta):
        loginit, logtrans, logobs = loglikhds

        zeta = alpha[:-1, :, None] + beta[1:, None, :] +\
               logobs[1:, :][:, None, :] + logtrans

        zeta -= zeta.max((1, 2))[:, None, None]
        zeta = np.exp(zeta)
        zeta /= zeta.sum((1, 2))[:, None, None]

        return zeta

    def viterbi(self, obs):
        loginit, logtrans, logobs = self.loglikhds(obs)
        T = logobs.shape[0]

        delta = np.zeros((T, self.nb_states))
        args = np.zeros((T, self.nb_states), np.int64)
        z = np.zeros((T, ), np.int64)

        aux = np.empty((self.nb_states,))
        for k in range(self.nb_states):
            aux[k] = logobs[0, k] + loginit[k]

        delta[0, :] = np.max(aux, axis=0)
        args[0, :] = np.argmax(delta[0, :], axis=0)

        for t in range(1, T):
            for j in range(self.nb_states):
                for i in range(self.nb_states):
                    aux[i] = delta[t - 1, i] + logtrans[t - 1, i, j] + logobs[t, j]

                delta[t, j] = np.max(aux, axis=0)
                args[t, j] = np.argmax(aux, axis=0)

        # backtrace
        z[T - 1] = np.argmax(delta[T - 1, :], axis=0)
        for t in range(T - 2, -1, -1):
            z[t] = args[t + 1, z[t + 1]]

        return delta, z

    def estep(self, obs):
        loglikhds = self.loglikhds(obs)
        alpha = self.forward(loglikhds)
        beta = self.backward(loglikhds)
        gamma = self.expected(alpha, beta)
        zeta = self.joint(loglikhds, alpha, beta)

        return gamma, zeta

    def mstep(self, obs, gamma, zeta):
        self.init_state.update(gamma[0, :])
        self.transitions.update(zeta, obs)
        self.observations.update(obs, gamma)

    def em(self, obs, nb_iter=50, prec=1e-6, verbose=False):
        lls = []
        last_ll = - np.inf

        it = 0
        while it < nb_iter:
            gamma, zeta = self.estep(obs)

            ll = self.logprob(obs)
            lls.append(ll)
            if verbose:
                print("it=", it, "ll=", ll)

            if (ll - last_ll) < prec:
                break
            else:
                self.mstep(obs, gamma, zeta)
                last_ll = ll

            it += 1

        return lls

    def permute(self, perm):
        self.init_state.permute(perm)
        self.transitions.permute(perm)
        self.observations.permute(perm)

    def lognorm(self, obs):
        loglikhds = self.loglikhds(obs)
        alpha = self.forward(loglikhds)
        return logsumexp(alpha[-1, :])

    def logprob(self, obs):
        return self.lognorm(obs) + self.logpriors()

    def smooth(self, obs):
        loglikhds = self.loglikhds(obs)
        alpha = self.forward(loglikhds)
        beta = self.backward(loglikhds)
        gamma = self.expected(alpha, beta)

        imu = np.array([self.init_observation.mu
                        for _ in range(self.nb_states)])
        armu = np.array([self.observations.mean(k, obs[:-1, :])
                         for k in range(self.nb_states)])

        return np.einsum('nk,knl->nl', gamma, np.concatenate((imu, armu), axis=1))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from hips.plotting.colormaps import gradient_cmap
    import seaborn as sns

    sns.set_style("white")
    sns.set_context("talk")

    color_names = [
        "windows blue",
        "red",
        "amber",
        "faded green",
        "dusty purple",
        "orange"
    ]

    colors = sns.xkcd_palette(color_names)
    cmap = gradient_cmap(colors)

    np.set_printoptions(precision=5, suppress=True)
    # npr.seed(1337)
    # np.seterr(all='raise')

    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    true_rarhmm = rARHMM(nb_states=3, dim_obs=2)

    T = 1500
    true_z, y = true_rarhmm.sample(T)
    true_ll = true_rarhmm.logprob(y)

    rarhmm = rARHMM(nb_states=3, dim_obs=2)
    rarhmm.initialize(y)

    lls = rarhmm.em(y, nb_iter=50, prec=1e-12, verbose=True)
    print("true_ll=", true_ll, "hmm_ll=", lls[-1])

    plt.figure(figsize=(5, 5))
    plt.plot(np.ones(len(lls)) * true_ll, '-r')
    plt.plot(lls)
    plt.show()

    rarhmm.permute(find_permutation(true_z, rarhmm.viterbi(y)[1]))
    _, rarhmm_z = rarhmm.viterbi(y)

    plt.figure(figsize=(8, 4))
    plt.subplot(211)
    plt.imshow(true_z[None, :], aspect="auto", cmap=cmap, vmin=0,
               vmax=len(colors) - 1)
    plt.xlim(0, T)
    plt.ylabel("$z_{\\mathrm{true}}$")
    plt.yticks([])

    plt.subplot(212)
    plt.imshow(rarhmm_z[None, :], aspect="auto", cmap=cmap, vmin=0,
               vmax=len(colors) - 1)
    plt.xlim(0, T)
    plt.ylabel("$z_{\\mathrm{inferred}}$")
    plt.yticks([])
    plt.xlabel("time")

    plt.tight_layout()
    plt.show()

    rarhmm_y = rarhmm.smooth(y)

    plt.figure(figsize=(8, 4))
    plt.plot(y + 10 * np.arange(rarhmm.dim_obs), '-k', lw=2)
    plt.plot(rarhmm_y + 10 * np.arange(rarhmm.dim_obs), '-', lw=2)
    plt.show()
