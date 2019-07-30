import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sds.rarhmm import rARHMM
from ssm.models import HMM as originHMM


T = [900, 950]

true_rarhmm = originHMM(3, 2, observations="ar", transitions="recurrent_only")

true_z, y = [], []
for t in T:
    _z, _y = true_rarhmm.sample(t)
    true_z.append(_z)
    y.append(_y)

true_ll = true_rarhmm.log_probability(y)

# true_rarhmm = rARHMM(nb_states=3, dim_obs=2)
# true_z, y = true_rarhmm.sample(T)
# true_ll = true_rarhmm.log_probability(y)

act = [np.zeros((t, 0)) for t in T]
my_rarhmm = rARHMM(nb_states=3, dim_obs=2)
my_rarhmm.initialize(y, act)
my_rarhmm_lls = my_rarhmm.em(y, act, nb_iter=50, prec=1e-12, verbose=False)

origin_rarhmm = originHMM(3, 2, observations="ar", transitions="recurrent_only")
origin_rarhmm_lls = origin_rarhmm.fit(y, method="em", num_em_iters=50, initialize=True, verbose=False)

print("true_ll=", true_ll, "my_rarhmm_ll=", my_rarhmm_lls[-1], "origin_rarhmm_ll=", origin_rarhmm_lls[-1])
