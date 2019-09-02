from sds import ARHMM

from sds.transitions import RecurrentTransition, NeuralRecurrentTransition
from sds.observations import AutoRegressiveGaussianObservation


class rARHMM(ARHMM):

    def __init__(self, nb_states, dm_obs, dm_act=0, type='recurrent'):
        super(rARHMM, self).__init__(nb_states, dm_obs, dm_act)

        self.type = type

        if self.type == 'recurrent':
            self.transitions = RecurrentTransition(self.nb_states, self.dm_obs,
                                                   self.dm_act, degree=3)
        elif self.type == 'neural-recurrent':
            self.transitions = NeuralRecurrentTransition(self.nb_states, self.dm_obs,
                                                         self.dm_act, hidden_layer_sizes=(10, ))

        self.observations = AutoRegressiveGaussianObservation(self.nb_states, self.dm_obs,
                                                              self.dm_act)
