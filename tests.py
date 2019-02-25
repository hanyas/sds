import numpy as np
import numpy.random as npr


# from sds.hmm_ls import HMM
# from ssm.models import HMM as linHMM
#
# true_hmm = HMM(nb_states=5, dim_obs=2)
#
# T = 200
# true_z, y = true_hmm.sample(T)
# true_ll = true_hmm.logprob(y)
#
# hmm = HMM(nb_states=5, dim_obs=2)
# hmm.initialize(y)
# hmm_lls = hmm.em(y, nb_iter=50, prec=1e-24, verbose=True)
#
# lhmm = linHMM(5, 2, observations="gaussian")
# lhmm_lls = lhmm.fit(y, method="em", initialize=True)
#
# print("true_ll=", true_ll, "hmm_ll=", hmm_lls[-1], "lhmm_ll=", lhmm_lls[-1])


from sds.arhmm_ls import ARHMM
from ssm.models import HMM as linHMM

true_arhmm = ARHMM(nb_states=3, dim_obs=2)

T = 900
true_z, y = true_arhmm.sample(T)
true_ll = true_arhmm.logprob(y)

arhmm = ARHMM(nb_states=3, dim_obs=2)
arhmm.initialize(y)
hmm_lls = arhmm.em(y, nb_iter=10, prec=1e-24, verbose=True)

lhmm = linHMM(3, 2, observations="ar")
lhmm_lls = lhmm.fit(y, method="em", initialize=True, verbose=True)

print("true_ll=", true_ll, "hmm_ll=", hmm_lls[-1], "lhmm_ll=", lhmm_lls[-1])
