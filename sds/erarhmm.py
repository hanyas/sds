import autograd.numpy as np

from sds import rARHMM

from sds.initial import GaussianInitControl
from sds.controls import AutoregRessiveLinearGaussianControl, LinearGaussianControl
from sds.utils import ensure_args_are_viable_lists


class erARHMM(rARHMM):

    def __init__(self, nb_states, dm_obs, dm_act, trans_type='neural',
                 init_state_prior={}, init_obs_prior={}, init_ctl_prior={}, trans_prior={}, obs_prior={}, ctl_prior={},
                 init_state_kwargs={}, init_obs_kwargs={}, init_ctl_kwargs={}, trans_kwargs={}, obs_kwargs={}, ctl_kwargs={},
                 learn_dyn=True, learn_ctl=False):

        super(erARHMM, self).__init__(nb_states, dm_obs, dm_act, trans_type,
                                      init_state_prior=init_state_prior, init_obs_prior=init_obs_prior, obs_prior=obs_prior, trans_prior=trans_prior,
                                      init_state_kwargs=init_state_kwargs, init_obs_kwargs=init_obs_kwargs, obs_kwargs=obs_kwargs, trans_kwargs=trans_kwargs)

        self.learn_dyn = learn_dyn
        self.learn_ctl = learn_ctl

        self.init_control = GaussianInitControl(self.nb_states, self.dm_obs, self.dm_act,
                                                prior=init_ctl_prior, **init_ctl_kwargs)

        # self.controls = LinearGaussianControl(self.nb_states, self.dm_obs, self.dm_act,
        #                                       prior=ctl_prior, **ctl_kwargs)

        self.controls = AutoregRessiveLinearGaussianControl(self.nb_states, self.dm_obs, self.dm_act,
                                                            prior=ctl_prior, **ctl_kwargs)

    @ensure_args_are_viable_lists
    def initialize(self, obs, act=None, **kwargs):
        super(erARHMM, self).initialize(obs, act, **kwargs)
        if self.learn_ctl:
            self.controls.initialize(obs, act, **kwargs)

    def log_priors(self):
        lp = 0.
        if self.learn_dyn:
            lp += super(erARHMM, self).log_priors()
        if self.learn_ctl:
            lp += self.controls.log_prior()
        return lp

    @ensure_args_are_viable_lists
    def log_likelihoods(self, obs, act=None):
        loginit, logtrans, logobs = super(erARHMM, self).log_likelihoods(obs, act)
        if self.learn_ctl:
            ilog = self.init_control.log_likelihood(obs, act)
            arlog = self.controls.log_likelihood(obs, act)

            logctl = []
            for _ilog, _arlog in zip(ilog, arlog):
                logctl.append(np.vstack((_ilog, _arlog)))

            return loginit, logtrans, logobs, logctl
        else:
            return loginit, logtrans, logobs

    def mstep(self, gamma, zeta,
              obs, act,
              init_mstep_kwargs,
              trans_mstep_kwargs,
              obs_mstep_kwargs, **kwargs):

        weights = kwargs.get('weights', None)
        if self.learn_dyn:
            self.init_observation.mstep(gamma, obs, act)
            self.init_control.mstep(gamma, obs, act)
            self.init_state.mstep(gamma, **init_mstep_kwargs)
            self.transitions.mstep(zeta, obs, act, weights, **trans_mstep_kwargs)
            self.observations.mstep(gamma, obs, act, weights, **obs_mstep_kwargs)
        if self.learn_ctl:
            ctl_mstep_kwargs = kwargs.get('ctl_mstep_kwargs', {})
            self.controls.mstep(gamma, obs, act, weights, **ctl_mstep_kwargs)

    def permute(self, perm):
        super(erARHMM, self).permute(perm)
        self.controls.permute(perm)

    @ensure_args_are_viable_lists
    def mean_control(self, obs, act=None):
        loglikhds = self.log_likelihoods(obs, act)
        alpha, norm = self.forward(*loglikhds)
        beta = self.backward(*loglikhds, scale=norm)
        gamma = self.posterior(alpha, beta)

        return self.controls.smooth(gamma, obs, act)
