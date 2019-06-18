import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sds.hmm_ls import HMM
from ssm.models import HMM as originHMM

T = [100, 95]

true_hmm = originHMM(5, 2, observations="gaussian")

true_z, y = [], []
for t in T:
    _z, _y = true_hmm.sample(t)
    true_z.append(_z)
    y.append(_y)

true_ll = true_hmm.log_probability(y)

# true_hmm = HMM(nb_states=5, dim_obs=2)
# true_z, y = true_hmm.sample(T)
# true_ll = true_hmm.log_probability(y)

my_hmm = HMM(nb_states=5, dim_obs=2)
my_hmm.initialize(y)
my_hmm_ll = my_hmm.em(y, nb_iter=50, prec=1e-12, verbose=False)

origin_hmm = originHMM(5, 2, observations="gaussian")
origin_hmm_ll = origin_hmm.fit(y, method="em", num_em_iters=50, initialize=True, verbose=False)

print("true_ll=", true_ll, "my_hmm_ll=", my_hmm_ll[-1], "origin_hmm_ll=", origin_hmm_ll[-1])
