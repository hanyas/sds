import autograd.numpy as np


from inf.sds.distributions import CategoricalInitState, GaussianObservation, StationaryTransition
from inf.sds.util import normalize, permutation


class HMM:

    def __init__(self, nb_states, dim_obs):
        self.nb_states = nb_states
        self.dim_obs = dim_obs

        # init state
        self.init_state = CategoricalInitState(self.nb_states)

        # transitions
        self.transitions = StationaryTransition(self.nb_states)

        # observations
        self.observations = GaussianObservation(self.nb_states, self.dim_obs)

        self.likhds = None

    def sample(self, T):
        obs = []
        state = []

        N = len(T)
        for n in range(N):
            _obs = np.zeros((T[n], self.dim_obs))
            _state = np.zeros((T[n],), np.int64)

            _state[0] = self.init_state.sample()
            for t in range(T[n] - 1):
                _obs[t, :] = self.observations.sample(_state[t])
                _state[t + 1] = self.transitions.sample(_state[t])

            _obs[-1, :] = self.observations.sample(_state[-1])

            state.append(_state)
            obs.append(_obs)

        return state, obs

    def initialize(self, obs):
        from sklearn.cluster import KMeans
        _obs = np.concatenate(obs)
        km = KMeans(self.nb_states).fit(_obs)

        self.observations.mu = km.cluster_centers_
        self.observations.cov = np.array([np.cov(_obs[km.labels_ == k].T)
                                          for k in range(self.nb_states)])

    def log_priors(self):
        logprior = 0.0
        logprior += self.init_state.log_prior()
        logprior += self.transitions.log_prior()
        logprior += self.observations.log_prior()
        return logprior

    def likelihoods(self, obs):
        likinit = self.init_state.likelihood()
        liktrans = self.transitions.likelihood()
        likobs = self.observations.likelihood(obs)
        return [likinit, liktrans, likobs]

    def filter(self, likhds):
        likinit, liktrans, likobs = likhds

        alpha = []
        norm = []
        for _likobs in likobs:
            T = _likobs.shape[0]
            _alpha = np.zeros((T, self.nb_states))
            _norm = np.zeros((T, 1))

            _alpha[0, :] = _likobs[0, :] * likinit
            _alpha[0, :], _norm[0, :] = normalize(_alpha[0, :], dim=0)

            for t in range(1, T):
                _alpha[t, :] = _likobs[t, :] * (liktrans.T @ _alpha[t - 1, :])
                _alpha[t, :], _norm[t, :] = normalize(_alpha[t, :], dim=0)

            alpha.append(_alpha)
            norm.append(_norm)

        return alpha, norm

    def smooth(self, likhds, scale):
        _, liktrans, likobs = likhds

        beta = []
        for _likobs, _scale in zip(likobs, scale):
            T = _likobs.shape[0]
            _beta = np.zeros((T, self.nb_states))

            _beta[-1, :] = np.ones((self.nb_states,)) / _scale[-1, None]
            for t in range(T - 2, -1, -1):
                _beta[t, :] = liktrans @ (_likobs[t + 1, :] * _beta[t + 1, :])
                _beta[t, :] = _beta[t, :] / _scale[t, None]

            beta.append(_beta)

        return beta

    def expectations(self, alpha, beta):
        return [normalize(_alpha * _beta, dim=1)[0] for _alpha, _beta in zip(alpha, beta)]

    def two_slice(self, likhds, alpha, beta):
        _, liktrans, likobs = likhds

        zeta = []
        for _likobs, _alpha, _beta in zip(likobs, alpha, beta):
            T = _likobs.shape[0]
            _zeta = np.zeros((T - 1, self.nb_states, self.nb_states))

            for t in range(T - 1):
                _zeta[t, :, :] = liktrans * np.outer(_alpha[t, :], _likobs[t + 1, :] * _beta[t + 1, :])
                _zeta[t, :, :], _ = normalize(_zeta[t, :, :], dim=(0, 1))

            zeta.append(_zeta)

        return zeta

    def viterbi(self, obs):
        likinit, liktrans, likobs = self.likelihoods(obs)

        delta = []
        z = []
        for _likobs in likobs:
            T = _likobs.shape[0]

            _delta = np.zeros((T, self.nb_states))
            _args = np.zeros((T, self.nb_states), np.int64)
            _z = np.zeros((T, ), np.int64)

            _aux = _likobs[0, :] * likinit
            _delta[0, :] = np.max(_aux, axis=0)
            _args[0, :] = np.argmax(_delta[0, :], axis=0)

            _delta[0, :], _ = normalize(_delta[0, :], dim=0)

            for t in range(1, T):
                for j in range(self.nb_states):
                    for i in range(self.nb_states):
                        _aux[i] = _delta[t - 1, i] * liktrans[i, j] * _likobs[t, j]

                    _delta[t, j] = np.max(_aux, axis=0)
                    _args[t, j] = np.argmax(_aux, axis=0)

                _delta[t, :], _ = normalize(_delta[t, :], dim=0)

            # backtrace
            _z[T - 1] = np.argmax(_delta[T - 1, :], axis=0)
            for t in range(T - 2, -1, -1):
                _z[t] = _args[t + 1, _z[t + 1]]

            delta.append(_delta)
            z.append(_z)

        return delta, z

    def estep(self, obs):
        self.likhds = self.likelihoods(obs)
        alpha, scale = self.filter(self.likhds)
        beta = self.smooth(self.likhds, scale)
        gamma = self.expectations(alpha, beta)
        zeta = self.two_slice(self.likhds, alpha, beta)

        return gamma, zeta

    def mstep(self, obs, gamma, zeta):
        self.init_state.mstep([_gamma[0, :] for _gamma in gamma])
        self.transitions.mstep(zeta)
        self.observations.mstep(obs, gamma)

    def em(self, obs, nb_iter=50, prec=1e-6, verbose=False):
        lls = []
        last_ll = - np.inf

        it = 0
        while it < nb_iter:
            gamma, zeta = self.estep(obs)

            ll = self.log_probability(obs)
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

    def log_norm(self, obs):
        if self.likhds is None:
            self.likhds = self.likelihoods(obs)
        _, norm = self.filter(self.likhds)
        return np.sum(np.log(np.concatenate(norm)))

    def log_probability(self, obs):
        return self.log_norm(obs) + self.log_priors()

    def mean_observation(self, obs):
        likhds = self.likelihoods(obs)
        alpha, scale = self.filter(likhds)
        beta = self.smooth(likhds, scale)
        gamma = self.expectations(alpha, beta)

        return [np.einsum('nk,km->nm', _gamma, self.observations.mu) for _gamma in gamma]


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

    true_hmm = HMM(nb_states=3, dim_obs=2)

    thetas = np.linspace(0, 2 * np.pi, true_hmm.nb_states, endpoint=False)
    for k in range(true_hmm.nb_states):
        true_hmm.observations.mu[k, :] = 3 * np.array([np.cos(thetas[k]), np.sin(thetas[k])])

    # trajectory lengths
    T = [95, 85, 75]

    true_z, y = true_hmm.sample(T=T)
    true_ll = true_hmm.log_probability(y)

    hmm = HMM(nb_states=3, dim_obs=2)
    hmm.initialize(y)

    lls = hmm.em(y, nb_iter=50, prec=1e-24, verbose=True)
    print("true_ll=", true_ll, "hmm_ll=", lls[-1])

    plt.figure(figsize=(5, 5))
    plt.plot(np.ones(len(lls)) * true_ll, '-r')
    plt.plot(lls)
    plt.show()

    _seq = np.random.choice(len(y))
    hmm.permute(permutation(true_z[_seq], hmm.viterbi([y[_seq]])[1][0]))
    _, hmm_z = hmm.viterbi([y[_seq]])

    plt.figure(figsize=(8, 4))
    plt.subplot(211)
    plt.imshow(true_z[_seq][None, :], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors) - 1)
    plt.xlim(0, len(y[_seq]))
    plt.ylabel("$z_{\\mathrm{true}}$")
    plt.yticks([])

    plt.subplot(212)
    plt.imshow(hmm_z[0][None, :], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors) - 1)
    plt.xlim(0, len(y[_seq]))
    plt.ylabel("$z_{\\mathrm{inferred}}$")
    plt.yticks([])
    plt.xlabel("time")

    plt.tight_layout()
    plt.show()

    hmm_y = hmm.mean_observation(y)

    plt.figure(figsize=(8, 4))
    plt.plot(y[_seq] + 10 * np.arange(hmm.dim_obs), '-k', lw=2)
    plt.plot(hmm_y[_seq] + 10 * np.arange(hmm.dim_obs), '-', lw=2)
    plt.show()
