import numpy as np
from sds_numpy.arhmm import ARHMM
from ssm.hmm import HMM as orgHMM

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
np.random.seed(1337)

T = [900, 950]

true_arhmm = orgHMM(3, 2, observations="ar")

true_z, x = [], []
for t in T:
    _z, _x = true_arhmm.sample(t)
    true_z.append(_z)
    x.append(_x)

true_ll = true_arhmm.log_probability(x)

# true_arhmm = ARHMM(nb_states=3, dm_obs=2)
# true_z, x = true_arhmm.sample(horizon=T)
# true_ll = true_arhmm.log_probability(x)

my_arhmm = ARHMM(nb_states=3, dm_obs=2)
my_arhmm.initialize(x)
my_ll = my_arhmm.em(x, nb_iter=1000, prec=0., verbose=True)

org_arhmm = orgHMM(3, 2, observations="ar")
org_ll = org_arhmm.fit(x, method="em", initialize=True)

print("true_ll=", true_ll, "my_ll=", my_ll[-1], "org_ll=", org_ll[-1])
