import numpy as np
from sds.rarhmm import rARHMM
from ssm.hmm import HMM as orgHMM

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
np.random.seed(1337)

T = [900, 950]

true_rarhmm = orgHMM(3, 2, observations="ar", transitions="recurrent")

true_z, x = [], []
for t in T:
    _z, _x = true_rarhmm.sample(t)
    true_z.append(_z)
    x.append(_x)

true_ll = true_rarhmm.log_probability(x)

# true_rarhmm = rARHMM(nb_states=3, dm_obs=2)
# true_z, x = true_rarhmm.sample(horizon=T)
# true_ll = true_rarhmm.log_probability(x)

my_rarhmm = rARHMM(nb_states=3, dm_obs=2, type='recurrent')
my_rarhmm.initialize(x)
my_ll = my_rarhmm.em(x, nb_iter=100, prec=1e-12)

org_rarhmm = orgHMM(3, 2, observations="ar", transitions="recurrent")
org_ll = org_rarhmm.fit(x, method="em", num_em_iters=100)

print("true_ll=", true_ll, "my_ll=", my_ll[-1], "org_ll=", org_ll[-1])
