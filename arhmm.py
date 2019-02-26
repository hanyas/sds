import autograd.numpy as np
import autograd.numpy.random as npr

from sds.distn import CategoricalInitState, StationaryTransition
from sds.distn import GaussianObservation, AutoRegressiveGaussianObservation

from sds.util import normalize, find_permutation
from sds.util import fit_linear_regression


class ARHMM:

    def __init__(self, nb_states, dim_obs):
        self.nb_states = nb_states
        self.dim_obs = dim_obs

        # init state
        self.init_state = CategoricalInitState(self.nb_states)

        # transitions
        self.transitions = StationaryTransition(self.nb_states)

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
            states[t] = self.transitions.sample(states[t - 1])
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

    def likhds(self, obs):
        likinit = self.init_state.lik()
        liktrans = self.transitions.lik()

        ilik = np.array([self.init_observation.lik(obs[0, :])
                         for _ in range(self.nb_states)]).T
        arlik = self.observations.lik(obs)
        likobs = np.concatenate((ilik, arlik))

        return [likinit, liktrans, likobs]

    def forward(self, likhds):
        likinit, liktrans, likobs = likhds
        T = likobs.shape[0]

        alpha = np.zeros((T, self.nb_states))
        norm = np.zeros((T, 1))

        alpha[0, :] = likobs[0, :] * likinit
        alpha[0, :], norm[0, :] = normalize(alpha[0, :], dim=0)

        for t in range(1, T):
            alpha[t, :] = likobs[t, :] * (liktrans.T @ alpha[t - 1, :])
            alpha[t, :], norm[t, :] = normalize(alpha[t, :], dim=0)

        return alpha, norm

    def backward(self, likhds, scale):
        _, liktrans, likobs = likhds
        T = likobs.shape[0]

        beta = np.zeros((T, self.nb_states))

        beta[-1, :] = np.ones((self.nb_states,)) / scale[-1, np.newaxis]
        for t in range(T - 2, -1, -1):
            beta[t, :] = liktrans @ (likobs[t + 1, :] * beta[t + 1, :])
            beta[t, :] = beta[t, :] / scale[t, np.newaxis]

        return beta

    def expected(self, alpha, beta):
        return normalize(alpha * beta, dim=1)[0]

    def joint(self, likhds, alpha, beta):
        _, liktrans, likobs = likhds
        T = likobs.shape[0]

        zeta = np.zeros((T - 1, self.nb_states, self.nb_states))

        for t in range(T - 1):
            zeta[t, :, :] = liktrans * np.outer(alpha[t, :], likobs[t + 1, :] * beta[t + 1, :])
            zeta[t, :, :], _ = normalize(zeta[t, :, :], dim=(0, 1))

        return zeta

    def viterbi(self, obs):
        likinit, liktrans, likobs = self.likhds(obs)
        T = likobs.shape[0]

        delta = np.zeros((T, self.nb_states))
        args = np.zeros((T, self.nb_states), np.int64)
        z = np.zeros((T, ), np.int64)

        aux = likobs[0, :] * likinit
        delta[0, :] = np.max(aux, axis=0)
        args[0, :] = np.argmax(delta[0, :], axis=0)

        delta[0, :], _ = normalize(delta[0, :], dim=0)

        for t in range(1, T):
            for j in range(self.nb_states):
                for i in range(self.nb_states):
                    aux[i] = delta[t - 1, i] * liktrans[i, j] * likobs[t, j]

                delta[t, j] = np.max(aux, axis=0)
                args[t, j] = np.argmax(aux, axis=0)

            delta[t, :], _ = normalize(delta[t, :], dim=0)

        # backtrace
        z[T - 1] = np.argmax(delta[T - 1, :], axis=0)
        for t in range(T - 2, -1, -1):
            z[t] = args[t + 1, z[t + 1]]

        return delta, z

    def estep(self, obs):
        likhds = self.likhds(obs)
        alpha, scale = self.forward(likhds)
        beta = self.backward(likhds, scale)
        gamma = self.expected(alpha, beta)
        zeta = self.joint(likhds, alpha, beta)

        return gamma, zeta

    def mstep(self, obs, gamma, zeta):
        self.init_state.update(gamma[0, :])
        self.transitions.update(zeta)
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
        likhds = self.likhds(obs)
        _, norm = self.forward(likhds)
        return np.sum(np.log(norm))

    def logprob(self, obs):
        return self.lognorm(obs) + self.logpriors()

    def smooth(self, obs):
        likhds = self.likhds(obs)
        alpha, scale = self.forward(likhds)
        beta = self.backward(likhds, scale)
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

    true_arhmm = ARHMM(nb_states=3, dim_obs=2)

    T = 900
    true_z, y = true_arhmm.sample(T)
    true_ll = true_arhmm.logprob(y)

    arhmm = ARHMM(nb_states=3, dim_obs=2)
    arhmm.initialize(y)

    lls = arhmm.em(y, nb_iter=50, prec=1e-24, verbose=True)
    print("true_ll=", true_ll, "hmm_ll=", lls[-1])

    plt.figure(figsize=(5, 5))
    plt.plot(np.ones(len(lls)) * true_ll, '-r')
    plt.plot(lls)
    plt.show()

    arhmm.permute(find_permutation(true_z, arhmm.viterbi(y)[1]))
    _, arhmm_z = arhmm.viterbi(y)

    plt.figure(figsize=(8, 4))
    plt.subplot(211)
    plt.imshow(true_z[None, :], aspect="auto", cmap=cmap, vmin=0,
               vmax=len(colors) - 1)
    plt.xlim(0, T)
    plt.ylabel("$z_{\\mathrm{true}}$")
    plt.yticks([])

    plt.subplot(212)
    plt.imshow(arhmm_z[None, :], aspect="auto", cmap=cmap, vmin=0,
               vmax=len(colors) - 1)
    plt.xlim(0, T)
    plt.ylabel("$z_{\\mathrm{inferred}}$")
    plt.yticks([])
    plt.xlabel("time")

    plt.tight_layout()
    plt.show()

    arhmm_y = arhmm.smooth(y)

    plt.figure(figsize=(8, 4))
    plt.plot(y + 10 * np.arange(arhmm.dim_obs), '-k', lw=2)
    plt.plot(arhmm_y + 10 * np.arange(arhmm.dim_obs), '-', lw=2)
    plt.show()
