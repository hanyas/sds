import numpy as np
import numpy.random as npr

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# from sds.hmm_ls import HMM
# from ssm.models import HMM as linHMM
#
# T = 200
#
# true_hmm = linHMM(5, 2, observations="gaussian")
# true_z, y = true_hmm.sample(T)
# true_ll = true_hmm.log_probability(y)
#
# # true_hmm = HMM(nb_states=5, dim_obs=2)
# # true_z, y = true_hmm.sample(T)
# # true_ll = true_hmm.logprob(y)
#
# hmm = HMM(nb_states=5, dim_obs=2)
# hmm.initialize(y)
# hmm_lls = hmm.em(y, nb_iter=50, prec=1e-24, verbose=True)
#
# lhmm = linHMM(5, 2, observations="gaussian")
# lhmm_lls = lhmm.fit(y, method="em", initialize=True)
#
# print("true_ll=", true_ll, "hmm_ll=", hmm_lls[-1], "lhmm_ll=", lhmm_lls[-1])
#

# from sds.arhmm_ls import ARHMM
# from ssm.models import HMM as linHMM
#
# T = 900
#
# true_arhmm = linHMM(3, 2, observations="ar")
# true_z, y = true_arhmm.sample(T)
# true_ll = true_arhmm.log_probability(y)
#
# # true_arhmm = ARHMM(nb_states=3, dim_obs=2)
# # true_z, y = true_arhmm.sample(T)
# # true_ll = true_arhmm.logprob(y)
#
# arhmm = ARHMM(nb_states=3, dim_obs=2)
# arhmm.initialize(y)
# hmm_lls = arhmm.em(y, nb_iter=50, prec=1e-24, verbose=True)
#
# lhmm = linHMM(3, 2, observations="ar")
# lhmm_lls = lhmm.fit(y, method="em", initialize=True, verbose=True)
#
# print("true_ll=", true_ll, "hmm_ll=", hmm_lls[-1], "lhmm_ll=", lhmm_lls[-1])


from sds.rarhmm_ls import rARHMM
from ssm.models import HMM as linHMM

T = 1500

true_rarhmm = linHMM(3, 2, observations="ar", transitions="recurrent_only")
true_z, y = true_rarhmm.sample(T)
true_ll = true_rarhmm.log_probability(y)

# true_rarhmm = rARHMM(nb_states=3, dim_obs=2)
# true_z, y = true_rarhmm.sample(T)
# true_ll = true_rarhmm.logprob(y)

rarhmm = rARHMM(nb_states=3, dim_obs=2)
rarhmm.initialize(y)
hmm_lls = rarhmm.em(y, nb_iter=50, prec=1e-24, verbose=True)

lhmm = linHMM(3, 2, observations="ar", transitions="recurrent_only")
lhmm_lls = lhmm.fit(y, method="em", initialize=True, verbose=False)

print("true_ll=", true_ll, "hmm_ll=", hmm_lls[-1], "lhmm_ll=", lhmm_lls[-1])
