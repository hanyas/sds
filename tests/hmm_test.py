import numpy as np
from sds.models.hmm import HMM as Ours
from ssm.hmm import HMM as Theirs

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
np.random.seed(1337)

T = [100, 95]

true_hmm = Theirs(5, 2, observations="gaussian")

true_z, x = [], []
for t in T:
    _z, _x = true_hmm.sample(t)
    true_z.append(_z)
    x.append(_x)

true_ll = true_hmm.log_probability(x)

# true_hmm = HMM(nb_states=5, obs_dim=2)
# true_z, x = true_hmm.sample(horizon=T)
# true_ll = true_hmm.log_probability(x)

our_hmm = Ours(nb_states=5, obs_dim=2)
our_hmm.initialize(x)
our_ll = our_hmm.em(x, nb_iter=1000, prec=0., verbose=True)

their_hmm = Theirs(5, 2, observations="gaussian")
their_ll = their_hmm.fit(x, method="em", num_em_iters=1000, initialize=True)

print("true_ll=", true_ll, "our_ll=", our_ll[-1], "their_ll=", their_ll[-1])
