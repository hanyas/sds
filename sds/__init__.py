from .hmm import HMM
from .arhmm import ARHMM
from .rarhmm import rARHMM
from .erarhmm import erARHMM

import os
import pickle
import torch

from gym.envs.registration import register

register(
    id='Pendulum-ID-v0',
    entry_point='sds.envs:Pendulum',
    max_episode_steps=1000,
)

register(
    id='Pendulum-ID-v1',
    entry_point='sds.envs:PendulumWithCartesianObservation',
    max_episode_steps=1000,
)

register(
    id='MassSpringDamper-ID-v0',
    entry_point='sds.envs:MassSpringDamper',
    max_episode_steps=1000,
)

register(
    id='Cartpole-ID-v0',
    entry_point='sds.envs:Cartpole',
    max_episode_steps=1000,
)

register(
    id='Cartpole-ID-v1',
    entry_point='sds.envs:CartpoleWithCartesianObservation',
    max_episode_steps=1000,
)

register(
    id='QQube-ID-v0',
    entry_point='sds.envs:QQube',
    max_episode_steps=1000,
    kwargs={'fs': 500.0, 'fs_ctrl': 100.0}
)

try:
    register(
        id='HybridMassSpringDamper-ID-v0',
        entry_point='sds.envs:HybridMassSpringDamper',
        max_episode_steps=1000,
        kwargs={'rarhmm': pickle.load(open(os.path.dirname(__file__)
                                           + '/envs/hybrid/models/poly_rarhmm_msd.pkl', 'rb'))}
    )
except:
    pass

try:
    register(
        id='HybridPendulum-ID-v0',
        entry_point='sds.envs:HybridPendulum',
        max_episode_steps=1000,
        kwargs={'rarhmm': torch.load(open(os.path.dirname(__file__)
                                          + '/envs/hybrid/models/neural_rarhmm_pendulum_polar.pkl', 'rb'),
                                     map_location='cpu')}
    )
except:
    pass

try:
    register(
        id='HybridPendulum-ID-v1',
        entry_point='sds.envs:HybridPendulumWithCartesianObservation',
        max_episode_steps=1000,
        kwargs={'rarhmm': torch.load(open(os.path.dirname(__file__)
                                          + '/envs/control/hybrid/models/local-poly_erarhmm_pendulum_cart.pkl', 'rb'),
                                     map_location='cpu')}
    )
except:
    pass
