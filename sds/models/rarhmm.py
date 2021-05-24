from sds.models import AutoRegressiveHiddenMarkovModel
from sds.transitions import SharedPolyOnlyTransition
from sds.transitions import SharedNeuralOnlyTransition
from sds.transitions import SharedPolyTransition
from sds.transitions import SharedNeuralTransition


class RecurrentAutoRegressiveHiddenMarkovModel(AutoRegressiveHiddenMarkovModel):

    def __init__(self, nb_states, obs_dim, act_dim=0, obs_lag=1,
                 algo_type='MAP', init_obs_type='full', trans_type='neural', obs_type='full',
                 init_state_prior={}, init_obs_prior={}, trans_prior={}, obs_prior={},
                 init_state_kwargs={}, init_obs_kwargs={}, trans_kwargs={}, obs_kwargs={}):

        super(RecurrentAutoRegressiveHiddenMarkovModel, self).__init__(nb_states, obs_dim, act_dim, obs_lag,
                                                                       algo_type, init_obs_type, obs_type,
                                                                       init_state_prior=init_state_prior,
                                                                       init_obs_prior=init_obs_prior,
                                                                       obs_prior=obs_prior,
                                                                       init_state_kwargs=init_state_kwargs,
                                                                       init_obs_kwargs=init_obs_kwargs,
                                                                       obs_kwargs=obs_kwargs)

        self.algo_type = algo_type
        self.init_obs_type = init_obs_type
        self.trans_type = trans_type
        self.obs_type = obs_type

        if self.trans_type == 'poly-only':
            self.transitions = SharedPolyOnlyTransition(self.nb_states, self.obs_dim, self.act_dim,
                                                        trans_prior, **trans_kwargs)
        elif self.trans_type == 'neural-only':
            self.transitions = SharedNeuralOnlyTransition(self.nb_states, self.obs_dim, self.act_dim,
                                                          trans_prior, **trans_kwargs)
        elif self.trans_type == 'poly':
            self.transitions = SharedPolyTransition(self.nb_states, self.obs_dim, self.act_dim,
                                                    trans_prior, **trans_kwargs)
        elif self.trans_type == 'neural':
            self.transitions = SharedNeuralTransition(self.nb_states, self.obs_dim, self.act_dim,
                                                      trans_prior, **trans_kwargs)
