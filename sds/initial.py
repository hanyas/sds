from autograd import numpy as np
import autograd.numpy.random as npr


class CategoricalInitState:

    def __init__(self, nb_states, reg=1e-8):
        self.nb_states = nb_states
        self.reg = reg

        self.pi = np.ones((self.nb_states, )) / self.nb_states

    def initialize(self, x, u):
        pass

    def sample(self):
        return npr.choice(self.nb_states, p=self.pi)

    def maximum(self):
        return np.argmax(self.pi)

    def log_init(self):
        return np.log(self.pi / self.pi.sum())

    def log_prior(self):
        return 0.0

    def permute(self, perm):
        self.pi = self.pi[perm]

    def mstep(self, w):
        self.pi = sum([_w for _w in w]) + self.reg
        self.pi /= np.sum(self.pi)
