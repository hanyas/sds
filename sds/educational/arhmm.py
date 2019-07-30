import autograd.numpy as np
import autograd.numpy.random as npr

from sds.transitions import StationaryTransition
from sds.initial import CategoricalInitState
from sds.observations import GaussianObservation, AutoRegressiveGaussianObservation

from sds.util import normalize, permutation, linear_regression


class ARHMM:

    def __init__(self, nb_states, dim_obs, dim_act=0):
        self.nb_states = nb_states
        self.dim_obs = dim_obs
        self.dim_act = dim_act

        # init state
        self.init_state = CategoricalInitState(self.nb_states)

        # transitions
        self.transitions = StationaryTransition(self.nb_states)

        # init observation
        self.init_observation = GaussianObservation(nb_states=1, dim_obs=self.dim_obs)

        # observations
        self.observations = AutoRegressiveGaussianObservation(self.nb_states, self.dim_obs, self.dim_act)

        self.likhds = None

    def sample(self, T, act):
        obs = []
        state = []

        N = len(T)
        for n in range(N):
            _act = act[n]
            _obs = np.zeros((T[n], self.dim_obs))
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
        self.init_observation.mu = npr.randn(1, self.dim_obs)
        self.init_observation.cov = np.array([np.eye(self.dim_obs, self.dim_obs)])

        Ts = [_obs.shape[0] for _obs in obs]
        if localize:
            from sklearn.cluster import KMeans
            km = KMeans(self.nb_states)
            km.fit(np.hstack((np.vstack(obs), np.vstack(act))))
            zs = np.split(km.labels_, np.cumsum(Ts)[:-1])
            zs = [z[:-1] for z in zs]
        else:
            zs = [npr.choice(self.nb_states, size=T - 1) for T in Ts]

        aux = np.zeros((self.nb_states, self.dim_obs, self.dim_obs))
        for k in range(self.nb_states):
            ts = [np.where(z == k)[0] for z in zs]
            xs = [np.hstack((_obs[t, :], _act[t, :])) for t, _obs, _act in zip(ts, obs, act)]
            ys = [_obs[t + 1, :] for t, _obs in zip(ts, obs)]

            coef_, intercept_, sigmas = linear_regression(xs, ys)
            self.observations.A[k, ...] = coef_[:, :self.dim_obs]
            self.observations.B[k, ...] = coef_[:, self.dim_obs:]
            self.observations.c[k, :] = intercept_
            aux[k, ...] = np.diag(sigmas)

        self.observations.cov = aux

    def log_priors(self):
        logprior = 0.0
        logprior += self.init_state.log_prior()
        logprior += self.transitions.log_prior()
        logprior += self.observations.log_prior()
        return logprior

    def likelihoods(self, obs, act):
        likinit = self.init_state.likelihood()
        liktrans = self.transitions.likelihood()

        ilik = self.init_observation.likelihood([_obs[0, :] for _obs in obs])
        arlik = self.observations.likelihood(obs, act)

        likobs = []
        for _ilik, _arlik in zip(ilik, arlik):
            likobs.append(np.vstack((np.repeat(_ilik, self.nb_states), _arlik)))

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

    def viterbi(self, obs, act):
        likinit, liktrans, likobs = self.likelihoods(obs, act)

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

    def estep(self, obs, act):
        self.likhds = self.likelihoods(obs, act)
        alpha, scale = self.filter(self.likhds)
        beta = self.smooth(self.likhds, scale)
        gamma = self.expectations(alpha, beta)
        zeta = self.two_slice(self.likhds, alpha, beta)

        return gamma, zeta

    def mstep(self, obs, act, gamma, zeta):
        self.init_state.mstep([_gamma[0, :] for _gamma in gamma])
        self.transitions.mstep(zeta)
        self.observations.mstep(obs, act, gamma)

    def em(self, obs, act, nb_iter=50, prec=1e-6, verbose=False):
        lls = []
        last_ll = - np.inf

        it = 0
        while it < nb_iter:
            gamma, zeta = self.estep(obs, act)

            ll = self.log_probability(obs, act)
            lls.append(ll)
            if verbose:
                print("it=", it, "ll=", ll)

            if (ll - last_ll) < prec:
                break
            else:
                self.mstep(obs, act, gamma, zeta)
                last_ll = ll

            it += 1

        return lls

    def permute(self, perm):
        self.init_state.permute(perm)
        self.transitions.permute(perm)
        self.observations.permute(perm)

    def log_norm(self, obs, act):
        if self.likhds is None:
            self.likhds = self.likelihoods(obs, act)
        _, norm = self.filter(self.likhds)
        return np.sum(np.log(np.concatenate(norm)))

    def log_probability(self, obs, act):
        return self.log_norm(obs, act) + self.log_priors()

    def mean_observation(self, obs, act):
        likhds = self.likelihoods(obs, act)
        alpha, scale = self.filter(likhds)
        beta = self.smooth(likhds, scale)
        gamma = self.expectations(alpha, beta)

        imu = np.array([self.init_observation.mu for _ in range(self.nb_states)])

        _mean = []
        for _obs, _act, _gamma in zip(obs, act, gamma):
            armu = np.array([self.observations.mean(k, _obs[:-1, :], _act[:-1, :self.dim_act]) for k in range(self.nb_states)])
            _mean.append(np.einsum('nk,knl->nl', _gamma, np.concatenate((imu, armu), axis=1)))

        return _mean


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

    true_arhmm = ARHMM(nb_states=3, dim_obs=2)

    # trajectory lengths
    T = [1250, 1150, 1025]

    # empty action sequence
    act = [np.zeros((t, 0)) for t in T]

    true_z, y = true_arhmm.sample(T=T, act=act)
    true_ll = true_arhmm.log_probability(y, act)

    arhmm = ARHMM(nb_states=3, dim_obs=2)
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
    plt.plot(y[_seq] + 10 * np.arange(arhmm.dim_obs), '-k', lw=2)
    plt.plot(arhmm_y[_seq] + 10 * np.arange(arhmm.dim_obs), '-', lw=2)
    plt.show()
