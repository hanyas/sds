import numpy as np
import torch

from sds_torch.hmm import HMM
from ssm.hmm import HMM as orgHMM

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
np.random.seed(1337)
torch.manual_seed(1337)

T = [100, 95]

true_hmm = orgHMM(5, 2, observations="gaussian")

true_z, x = [], []
for t in T:
    _z, _x = true_hmm.sample(t)
    true_z.append(_z)
    x.append(torch.from_numpy(_x))

# true_ll = true_hmm.log_probability(x)

# true_hmm = HMM(nb_states=5, dm_obs=2)
# true_z, x = true_hmm.sample(horizon=T)
# true_ll = true_hmm.log_probability(x)

my_hmm = HMM(nb_states=5, dm_obs=2)
my_hmm.initialize(x)
my_ll = my_hmm.em(x, nb_iter=1000, prec=0., verbose=True)

# org_hmm = orgHMM(5, 2, observations="gaussian")
# org_ll = org_hmm.fit(x, method="em", initialize=True)
#
# print("true_ll=", true_ll, "my_ll=", my_ll[-1], "org_ll=", org_ll[-1])
