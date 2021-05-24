import numpy as np
import numpy.random as npr

import torch

from sds.models import RecurrentAutoRegressiveHiddenMarkovModel
from sds.utils.general import random_rotation

import matplotlib.pyplot as plt

# npr.seed(1337)
# torch.manual_seed(1337)
torch.set_num_threads(1)


# from https://github.com/lindermanlab/ssm
def make_nascar_model():
    As = [random_rotation(2, np.pi/24.),
          random_rotation(2, np.pi/48.)]

    # Set the center points for each system
    centers = [np.array([+2.0, 0.]),
               np.array([-2.0, 0.])]
    cs = [-(A - np.eye(2)).dot(center) for A, center in zip(As, centers)]

    # Add a "right" state
    As.append(np.eye(2))
    cs.append(np.array([+0.1, 0.]))

    # Add a "right" state
    As.append(np.eye(2))
    cs.append(np.array([-0.25, 0.]))

    # Construct multinomial regression to divvy up the space
    w1 = 100 * np.array([-2.0, +1.0, 0.0])   # x + b > 0 -> x > -b
    w2 = 100 * np.array([-2.0, -1.0, 0.0])   # -x + b > 0 -> x < b
    w3 = 10 * np.array([0.0, 0.0, +1.0])    # y > 0
    w4 = 10 * np.array([0.0, 0.0, -1.0])    # y < 0
    coef = np.row_stack((w1, w2, w3, w4))

    true_rarhmm = RecurrentAutoRegressiveHiddenMarkovModel(nb_states=4, obs_dim=2,
                                                           trans_type='poly-only')

    true_rarhmm.init_observation.mu = np.tile(np.array([[0, 1]]), (4, 1))
    true_rarhmm.init_observation.sigma = np.array([1e0 * np.eye(2) for _ in range(4)])
    true_rarhmm.observations.A = np.array(As)
    true_rarhmm.observations.c = np.array(cs)
    true_rarhmm.observations.sigma = np.array([1e-4 * np.eye(2) for _ in range(4)])

    true_rarhmm.transitions.params = coef

    return true_rarhmm


true_rarhmm = make_nascar_model()

# trajectory lengths
T = [750, 750, 750]

true_z, x = true_rarhmm.sample(horizon=T)
true_ll = true_rarhmm.log_normalizer(x)

# poly transition
trans_type = 'poly-only'
trans_kwargs = {'norm': {'mean': np.mean(np.vstack(x), axis=0),
                         'std': np.std(np.vstack(x), axis=0)},
                'device': 'cpu'}

# # neural transition
# trans_type = 'neural-only'
# trans_kwargs = {'hidden_sizes': (16, ), 'activation': 'relu',
#                 'norm': {'mean': np.mean(np.vstack(x), axis=0),
#                          'std': np.std(np.vstack(x), axis=0)},
#                 'device': 'cpu'}

trans_mstep_kwargs = {'nb_iter': 50, 'l2': 1e-32}

# npr.seed(1337)
std_rarhmm = RecurrentAutoRegressiveHiddenMarkovModel(nb_states=4, obs_dim=2,
                                                      algo_type='MAP',
                                                      trans_type=trans_type,
                                                      trans_kwargs=trans_kwargs)

std_lls = std_rarhmm.em(x, nb_iter=1000,
                        prec=0., initialize=True,
                        trans_mstep_kwargs=trans_mstep_kwargs)

print("true_ll=", true_ll, "std_ll=", std_lls[-1])

plt.figure(figsize=(7, 7))
plt.axhline(y=true_ll, color='r')
plt.plot(std_lls)
plt.xscale('symlog')
plt.yscale('symlog')
plt.show()

seq = npr.choice(len(x))
std_rarhmm.plot(x[seq], true_state=true_z[seq], title='Standard')
