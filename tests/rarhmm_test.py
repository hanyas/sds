from sds.models.rarhmm import RecurrentAutoRegressiveHiddenMarkovModel as Ours
from ssm.hmm import HMM as Theirs

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
# np.random.seed(1337)

T = [900, 950]

true_rarhmm = Theirs(3, 2, observations="ar", transitions="recurrent")

true_z, x = [], []
for t in T:
    _z, _x = true_rarhmm.sample(t)
    true_z.append(_z)
    x.append(_x)

true_ll = true_rarhmm.log_probability(x)

# true_rarhmm = rARHMM(nb_states=3, obs_dim=2)
# true_z, x = true_rarhmm.sample(horizon=T)
# true_ll = true_rarhmm.log_probability(x)

our_rarhmm = Ours(nb_states=3, obs_dim=2, trans_type='recurrent')
our_rarhmm.initialize(x)
our_ll = our_rarhmm.em(x, nb_iter=100, prec=1e-12)

their_rarhmm = Theirs(3, 2, observations="ar", transitions="recurrent")
their_ll = their_rarhmm.fit(x, method="em", num_em_iters=100)

print("true_ll=", true_ll, "our_ll=", our_ll[-1], "their_ll=", their_ll[-1])
