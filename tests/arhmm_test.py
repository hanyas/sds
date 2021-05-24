import numpy as np
from sds.models.arhmm import AutoRegressiveHiddenMarkovModel as Ours
from ssm.hmm import HMM as Theirs

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
np.random.seed(1337)

T = [900, 950]

true_arhmm = Theirs(3, 2, observations="ar")

true_z, x = [], []
for t in T:
    _z, _x = true_arhmm.sample(t)
    true_z.append(_z)
    x.append(_x)

true_ll = true_arhmm.log_probability(x)

# true_arhmm = ARHMM(nb_states=3, obs_dim=2)
# true_z, x = true_arhmm.sample(horizon=T)
# true_ll = true_arhmm.log_probability(x)

our_arhmm = Ours(nb_states=3, obs_dim=2)
our_arhmm.initialize(x)
our_ll = our_arhmm.em(x, nb_iter=1000, prec=0., verbose=True)

their_arhmm = Theirs(3, 2, observations="ar")
their_ll = their_arhmm.fit(x, method="em", num_em_iters=1000, initialize=True)

print("true_ll=", true_ll, "our_ll=", our_ll[-1], "their_ll=", their_ll[-1])
