from autograd import numpy as np
import autograd.numpy.random as npr

from autograd.scipy.special import logsumexp


class CategoricalInitState:

    def __init__(self, nb_states, prior, reg=1e-16):
        self.nb_states = nb_states

        self.prior = prior
        self.reg = reg

        self.log_pi = -np.log(self.nb_states) * np.ones(self.nb_states)

    @property
    def params(self):
        return tuple([self.log_pi])

    @params.setter
    def params(self, value):
        self.log_pi = value[0]

    @property
    def pi(self):
        return np.exp(self.log_pi - logsumexp(self.log_pi))

    def initialize(self, x, u):
        pass

    def sample(self):
        return npr.choice(self.nb_states, p=self.pi)

    def maximum(self):
        return np.argmax(self.pi)

    def log_init(self):
        return self.log_pi - logsumexp(self.log_pi)

    def log_prior(self):
        lp = 0.
        if self.prior:
            pass
        return lp

    def permute(self, perm):
        self.log_pi = self.log_pi[perm]

    def mstep(self, w, **kwargs):
        _pi = sum([_w for _w in w]) + self.reg
        self.log_pi = np.log(_pi / sum(_pi))
