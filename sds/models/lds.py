import numpy as np
import numpy.random as npr

from sds.initial import SingleBayesianInitGaussianLatent
from sds.latents import SingleBayesianAutoRegressiveGaussianLatent
from sds.emissions import SingleBayesianLinearGaussianEmission

from sds.utils.decorate import ensure_args_are_viable
from sds.utils.general import Statistics as Stats

from operator import add
from functools import reduce

from tqdm import trange


class LinearGaussianDynamics:

    def __init__(self, ems_dim, act_dim, ltn_dim, ltn_lag=1,
                 init_ltn_prior=None, ltn_prior=None, ems_prior=None,
                 init_ltn_kwargs={}, ltn_kwargs={}, ems_kwargs={}):

        self.ltn_dim = ltn_dim
        self.act_dim = act_dim
        self.ems_dim = ems_dim

        self.latent_lag = ltn_lag

        self.init_ltn_prior = init_ltn_prior
        self.ltn_prior = ltn_prior
        self.ems_prior = ems_prior

        self.init_latent = SingleBayesianInitGaussianLatent(self.ltn_dim, self.act_dim, self.latent_lag,
                                                            self.init_ltn_prior, **init_ltn_kwargs)

        self.latent = SingleBayesianAutoRegressiveGaussianLatent(self.ltn_dim, self.act_dim, self.latent_lag,
                                                                 self.ltn_prior, **ltn_kwargs)

        self.emission = SingleBayesianLinearGaussianEmission(self.ltn_dim, self.ems_dim,
                                                             self.ems_prior, **ems_kwargs)

    @property
    def params(self):
        return self.init_latent.params,\
               self.latent.params,\
               self.emission.params

    @params.setter
    def params(self, value):
        self.init_latent.params = value[0]
        self.latent.params = value[1]
        self.emission.params = value[2]

    @ensure_args_are_viable
    def initialize(self, ems, act=None, **kwargs):
        pass

    @ensure_args_are_viable
    def kalman_filter(self, ems, act=None):
        if isinstance(ems, np.ndarray) \
                and isinstance(act, np.ndarray):

            nb_steps = ems.shape[0]
            filt_mean = np.zeros((nb_steps, self.ltn_dim))
            filt_covar = np.zeros((nb_steps, self.ltn_dim, self.ltn_dim))

            mu, lmbda = self.init_latent.likelihood.params
            pred_mean, pred_cov = mu, np.linalg.inv(lmbda)

            loglik = 0.
            for t in range(nb_steps):
                loglik += self.emission.expected_log_liklihood(pred_mean, pred_cov, ems[t])  # loglik
                filt_mean[t], filt_covar[t] = self.emission.condition(pred_mean, pred_cov, ems[t])    # condition
                pred_mean, pred_cov = self.latent.propagate(filt_mean[t], filt_covar[t], act[t])  # predict

            return filt_mean, filt_covar, loglik
        else:
            def inner(ems, act):
                return self.kalman_filter.__wrapped__(self, ems, act)
            result = map(inner, ems, act)
            filt_mean, filt_covar, loglik = list(map(list, zip(*result)))
            return filt_mean, filt_covar, np.sum(np.hstack(loglik))

    @ensure_args_are_viable
    def kalman_smoother(self, ems, act=None):
        if isinstance(ems, np.ndarray) \
                and isinstance(act, np.ndarray):

            filt_mean, filt_covar, loglik = self.kalman_filter(ems, act)

            nb_steps = ems.shape[0]
            smooth_mean = np.zeros((nb_steps, self.ltn_dim))
            smooth_covar = np.zeros((nb_steps, self.ltn_dim, self.ltn_dim))

            gain = np.zeros((nb_steps - 1, self.ltn_dim, self.ltn_dim))

            smooth_mean[-1], smooth_covar[-1] = filt_mean[-1], filt_covar[-1]
            for t in range(nb_steps - 2, -1, -1):
                smooth_mean[t], smooth_covar[t], gain[t] =\
                    self.latent.smooth(smooth_mean[t + 1], smooth_covar[t + 1],
                                       filt_mean[t], filt_covar[t], act[t])

            return smooth_mean, smooth_covar, gain, loglik
        else:
            def inner(ems, act):
                return self.kalman_smoother.__wrapped__(self, ems, act)
            result = map(inner, ems, act)
            smooth_mean, smooth_covar, gain, loglik = list(map(list, zip(*result)))
            return smooth_mean, smooth_covar, gain, np.sum(np.hstack(loglik))

    @ensure_args_are_viable
    def estep(self, ems, act=None):
        if isinstance(ems, np.ndarray) \
                and isinstance(act, np.ndarray):

            smooth_mean, smooth_covar, gain, loglik =\
                self.kalman_smoother(ems, act)

            nb_steps = ems.shape[0]

            # currently only for full covariances
            Ex = smooth_mean  # E[x{n}]
            ExxpT = np.zeros_like(gain)  # E[x_{n} x_{n-1}^T]
            for t in range(nb_steps - 1):
                ExxpT[t] = smooth_covar[t + 1] @ gain[t].T\
                           + np.outer(smooth_mean[t + 1], smooth_mean[t])

            ExxT = np.zeros_like(smooth_covar)  # E[x_{n} x_{n}^T]
            for t in range(nb_steps):
                ExxT[t] = smooth_covar[t] + np.outer(smooth_mean[t], smooth_mean[t])

            # init_ltn_stats
            x, xxT = Ex[0], ExxT[0]
            init_ltn_stats = Stats([x, 1., xxT, 1.])

            # ltn_stats
            xxT = np.zeros((nb_steps - 1, self.ltn_dim + 1, self.ltn_dim + 1))
            for t in range(nb_steps - 1):
                xxT[t] = np.block([[ExxT[t], Ex[t][:, np.newaxis]],
                                   [Ex[t][np.newaxis, :], np.ones((1,))]])

            yxT = np.zeros((nb_steps - 1, self.ltn_dim, self.ltn_dim + 1))
            for t in range(nb_steps - 1):
                yxT[t] = np.hstack((ExxpT[t], Ex[t + 1][:, np.newaxis]))

            yyT = ExxT[1:]

            ltn_stats = Stats([np.sum(yxT, axis=0),
                               np.sum(xxT, axis=0),
                               np.sum(yyT, axis=0),
                               yyT.shape[0]])

            # ems_stats
            xxT = np.zeros((nb_steps, self.ltn_dim + 1, self.ltn_dim + 1))
            for t in range(nb_steps):
                xxT[t] = np.block([[ExxT[t], Ex[t][:, np.newaxis]],
                                   [Ex[t][np.newaxis, :], np.ones((1,))]])

            x = np.hstack((Ex, np.ones((Ex.shape[0], 1))))
            yxT = np.einsum('nd,nl->ndl', ems, x)
            yyT = np.einsum('nd,nl->ndl', ems, ems)

            ems_stats = Stats([np.sum(yxT, axis=0),
                               np.sum(xxT, axis=0),
                               np.sum(yyT, axis=0),
                               yyT.shape[0]])

            return init_ltn_stats, ltn_stats, ems_stats, loglik
        else:
            def inner(ems, act):
                return self.estep.__wrapped__(self, ems, act)
            result = map(inner, ems, act)
            init_ltn_stats, ltn_stats, ems_stats, loglik = list(map(list, zip(*result)))
            stats = tuple([reduce(add, init_ltn_stats),
                           reduce(add, ltn_stats),
                           reduce(add, ems_stats)])
            return stats, np.sum(np.hstack(loglik))

    def mstep(self, stats, ems, act,
              init_ltn_mstep_kwarg,
              ltn_mstep_kwarg,
              ems_mstep_kwargs):

        init_ltn_stats, ltn_stats, ems_stats = stats

        self.init_latent.mstep(init_ltn_stats, **init_ltn_mstep_kwarg)
        self.latent.mstep(ltn_stats, **ltn_mstep_kwarg)
        # self.emission.mstep(ems_stats, **ems_mstep_kwargs)

    def em(self, train_ems, train_act=None,
           nb_iter=50, prec=1e-4, initialize=True,
           init_ltn_mstep_kwarg={},
           ltn_mstep_kwarg={},
           ems_mstep_kwarg={}, **kwargs):

        proc_id = kwargs.pop('proc_id', 0)

        if initialize:
            self.initialize(train_ems, train_act)

        train_lls = []
        stats, train_ll = self.estep(train_ems, train_act)

        train_ll += self.init_latent.prior.log_likelihood(self.init_latent.likelihood.params)[0]
        train_ll += self.latent.prior.log_likelihood(self.latent.likelihood.params)[0]
        # train_ll += self.emission.prior.log_likelihood(self.emission.likelihood.params)[0]

        train_lls.append(train_ll)
        last_train_ll = train_ll

        pbar = trange(nb_iter, position=proc_id)
        pbar.set_description("#{}, ll: {:.5f}".format(proc_id, train_lls[-1]))

        for _ in pbar:
            self.mstep(stats,
                       train_ems, train_act,
                       init_ltn_mstep_kwarg,
                       ltn_mstep_kwarg,
                       ems_mstep_kwarg)

            stats, train_ll = self.estep(train_ems, train_act)

            train_ll += self.init_latent.prior.log_likelihood(self.init_latent.likelihood.params)[0]
            train_ll += self.latent.prior.log_likelihood(self.latent.likelihood.params)[0]
            # train_ll += self.emission.prior.log_likelihood(self.emission.likelihood.params)[0]

            train_lls.append(train_ll)

            pbar.set_description("#{}, ll: {:.5f}".format(proc_id, train_lls[-1]))

            if abs(train_ll - last_train_ll) < prec:
                break
            else:
                last_train_ll = train_ll

        return train_lls
