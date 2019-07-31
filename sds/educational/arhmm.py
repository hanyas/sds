import autograd.numpy as np
import autograd.numpy.random as npr

from sds.observations import GaussianObservation, AutoRegressiveGaussianObservation
from sds.utils import permutation, linear_regression

from sds.educational.hmm import HMM


class ARHMM(HMM):

    def __init__(self, nb_states, dm_obs, dm_act=0):
        super(ARHMM, self).__init__(nb_states, dm_obs, dm_act)
        # init observation
        self.init_observation = GaussianObservation(nb_states=1, dm_obs=self.dm_obs, dm_act=dm_act)

        # observations
        self.observations = AutoRegressiveGaussianObservation(self.nb_states, self.dm_obs, self.dm_act)

        self.likhds = None

    def sample(self, T, act):
        obs = []
        state = []

        N = len(T)
        for n in range(N):
            _act = act[n]
            _obs = np.zeros((T[n], self.dm_obs))
            _state = np.zeros((T[n], ), np.int64)

            _state[0] = self.init_state.sample()
            _obs[0, :] = self.init_observation.sample(z=0)
            for t in range(1, T[n]):
                _state[t] = self.transitions.sample(_state[t - 1])
                _obs[t, :] = self.observations.sample(_state[t], _obs[t - 1, :], _act[t - 1, :])

            state.append(_state)
            obs.append(_obs)

        return state, obs

    def initialize(self, obs, act, localize=True):
        self.init_observation.mu = npr.randn(1, self.dm_obs)
        self.init_observation.cov = np.array([np.eye(self.dm_obs, self.dm_obs)])

        Ts = [_obs.shape[0] for _obs in obs]
        if localize:
            from sklearn.cluster import KMeans
            km = KMeans(self.nb_states)
            km.fit(np.hstack((np.vstack(obs), np.vstack(act))))
            zs = np.split(km.labels_, np.cumsum(Ts)[:-1])
            zs = [z[:-1] for z in zs]
        else:
            zs = [npr.choice(self.nb_states, size=T - 1) for T in Ts]

        aux = np.zeros((self.nb_states, self.dm_obs, self.dm_obs))
        for k in range(self.nb_states):
            ts = [np.where(z == k)[0] for z in zs]
            xs = [np.hstack((_obs[t, :], _act[t, :])) for t, _obs, _act in zip(ts, obs, act)]
            ys = [_obs[t + 1, :] for t, _obs in zip(ts, obs)]

            coef_, intercept_, sigmas = linear_regression(xs, ys)
            self.observations.A[k, ...] = coef_[:, :self.dm_obs]
            self.observations.B[k, ...] = coef_[:, self.dm_obs:]
            self.observations.c[k, :] = intercept_
            aux[k, ...] = np.diag(sigmas)

        self.observations.cov = aux

    def likelihoods(self, obs, act):
        likinit = self.init_state.likelihood()
        liktrans = self.transitions.likelihood(obs, act)

        ilik = self.init_observation.likelihood([_obs[0, :] for _obs in obs])
        arlik = self.observations.likelihood(obs, act)

        likobs = []
        for _ilik, _arlik in zip(ilik, arlik):
            likobs.append(np.vstack((np.repeat(_ilik, self.nb_states), _arlik)))

        return [likinit, liktrans, likobs]

    def mean_observation(self, obs, act):
        likhds = self.likelihoods(obs, act)
        alpha, scale = self.filter(likhds)
        beta = self.smooth(likhds, scale)
        gamma = self.marginals(alpha, beta)

        imu = np.array([self.init_observation.mu for _ in range(self.nb_states)])

        _mean = []
        for _obs, _act, _gamma in zip(obs, act, gamma):
            armu = np.array([self.observations.mean(k, _obs[:-1, :], _act[:-1, :self.dm_act]) for k in range(self.nb_states)])
            _mean.append(np.einsum('nk,knl->nl', _gamma, np.concatenate((imu, armu), axis=1)))

        return _mean


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)

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

    true_arhmm = ARHMM(nb_states=3, dm_obs=2, dm_act=0)

    # trajectory lengths
    T = [1250, 1150, 1025]

    # empty action sequence
    act = [np.zeros((t, 0)) for t in T]

    true_z, y = true_arhmm.sample(T=T, act=act)
    true_ll = true_arhmm.log_probability(y, act)

    arhmm = ARHMM(nb_states=3, dm_obs=2, dm_act=0)
    arhmm.initialize(y, act)

    lls = arhmm.em(y, act, nb_iter=50, prec=1e-6, verbose=True)
    print("true_ll=", true_ll, "hmm_ll=", lls[-1])

    plt.figure(figsize=(5, 5))
    plt.plot(np.ones(len(lls)) * true_ll, '-r')
    plt.plot(lls)
    plt.show()

    _seq = np.random.choice(len(y))
    arhmm.permute(permutation(true_z[_seq], arhmm.viterbi([y[_seq]], [act[_seq]])[1][0]))
    _, arhmm_z = arhmm.viterbi([y[_seq]], [act[_seq]])

    plt.figure(figsize=(8, 4))
    plt.subplot(211)
    plt.imshow(true_z[_seq][None, :], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors) - 1)
    plt.xlim(0, len(y[_seq]))
    plt.ylabel("$z_{\\mathrm{true}}$")
    plt.yticks([])

    plt.subplot(212)
    plt.imshow(arhmm_z[0][None, :], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors) - 1)
    plt.xlim(0, len(y[_seq]))
    plt.ylabel("$z_{\\mathrm{inferred}}$")
    plt.yticks([])
    plt.xlabel("time")

    plt.tight_layout()
    plt.show()

    arhmm_y = arhmm.mean_observation(y, act)

    plt.figure(figsize=(8, 4))
    plt.plot(y[_seq] + 10 * np.arange(arhmm.dm_obs), '-k', lw=2)
    plt.plot(arhmm_y[_seq] + 10 * np.arange(arhmm.dm_obs), '-', lw=2)
    plt.show()
