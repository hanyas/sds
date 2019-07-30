#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: initial
# @Date: 2019-07-30-15-29
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de
from autograd import numpy as np
from scipy.stats import multinomial as cat


class CategoricalInitState:

    def __init__(self, nb_states, reg=1e-8):
        self.nb_states = nb_states
        self.reg = reg

        self.pi = np.ones((self.nb_states, )) / self.nb_states

    def sample(self):
        return np.argmax(cat(1, self.pi).rvs())

    def likelihood(self):
        return self.pi

    def log_likelihood(self):
        return np.log(self.likelihood())

    def log_prior(self):
        return 0.0

    def permute(self, perm):
        self.pi = self.pi[perm]

    def mstep(self, w):
        self.pi = sum([_w for _w in w]) + self.reg
        self.pi /= np.sum(self.pi)