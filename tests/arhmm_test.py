import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sds.arhmm import ARHMM
from ssm.models import HMM as originHMM


T = [900, 950]

true_arhmm = originHMM(3, 2, observations="ar")

true_z, y = [], []
for t in T:
    _z, _y = true_arhmm.sample(t)
    true_z.append(_z)
    y.append(_y)

true_ll = true_arhmm.log_probability(y)

# true_arhmm = ARHMM(nb_states=3, dm_obs=2)
# true_z, y = true_arhmm.sample(T)
# true_ll = true_arhmm.log_probability(y)

act = [np.zeros((t, 0)) for t in T]
my_arhmm = ARHMM(nb_states=3, dm_obs=2)
my_arhmm.initialize(y, act)
my_arhmm_lls = my_arhmm.em(y, act, nb_iter=50, prec=1e-12, verbose=False)

originarhmm = originHMM(3, 2, observations="ar")
origin_arhmm_ll = originarhmm.fit(y, method="em", num_em_iters=50, initialize=True, verbose=False)

print("true_ll=", true_ll, "my_arhmm_ll=", my_arhmm_lls[-1], "origin_arhmm_ll=", origin_arhmm_ll[-1])
