from sds import ARHMM

from sds.transitions import PolyRecurrentTransition
from sds.transitions import NeuralRecurrentTransition


class rARHMM(ARHMM):

    def __init__(self, nb_states, dm_obs, dm_act=0, trans_type='neural',
                 init_state_prior={}, init_obs_prior={}, trans_prior={}, obs_prior={},
                 init_state_kwargs={}, init_obs_kwargs={}, trans_kwargs={}, obs_kwargs={}):

        super(rARHMM, self).__init__(nb_states, dm_obs, dm_act,
                                     init_state_prior=init_state_prior, init_obs_prior=init_obs_prior, obs_prior=obs_prior,
                                     init_state_kwargs=init_state_kwargs, init_obs_kwargs=init_obs_kwargs, obs_kwargs=obs_kwargs)

        self.trans_type = trans_type

        if self.trans_type == 'poly':
            self.transitions = PolyRecurrentTransition(self.nb_states, self.dm_obs, self.dm_act,
                                                       prior=trans_prior, **trans_kwargs)
        elif self.trans_type == 'neural':
            self.transitions = NeuralRecurrentTransition(self.nb_states, self.dm_obs, self.dm_act,
                                                         prior=trans_prior, **trans_kwargs)
