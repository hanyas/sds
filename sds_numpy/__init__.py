from .hmm import HMM
from .arhmm import ARHMM
from .rarhmm import rARHMM
from .erarhmm import erARHMM
from .ensemble import Ensemble

import os
import torch

from gym.envs.registration import register

register(
    id='MassSpringDamper-ID-v0',
    entry_point='sds_numpy.envs:MassSpringDamper',
    max_episode_steps=1000,
)

register(
    id='BouncingBall-ID-v0',
    entry_point='sds_numpy.envs:BouncingBall',
    max_episode_steps=1000,
)

register(
    id='Pendulum-ID-v0',
    entry_point='sds_numpy.envs:Pendulum',
    max_episode_steps=1000,
)

register(
    id='Pendulum-ID-v1',
    entry_point='sds_numpy.envs:PendulumWithCartesianObservation',
    max_episode_steps=1000,
)

register(
    id='Cartpole-ID-v0',
    entry_point='sds_numpy.envs:Cartpole',
    max_episode_steps=1000,
)

register(
    id='Cartpole-ID-v1',
    entry_point='sds_numpy.envs:CartpoleWithCartesianObservation',
    max_episode_steps=1000,
)

register(
    id='QQube-ID-v0',
    entry_point='sds_numpy.envs:Qube',
    max_episode_steps=1000,
    kwargs={'fs': 500.0, 'fs_ctrl': 100.0}
)

register(
    id='QQube-ID-v1',
    entry_point='sds_numpy.envs:QubeWithCartesianObservation',
    max_episode_steps=1000,
    kwargs={'fs': 500.0, 'fs_ctrl': 100.0}
)

try:
    register(
        id='HybridMassSpringDamper-ID-v0',
        entry_point='sds_numpy.envs:HybridMassSpringDamper',
        max_episode_steps=1000,
        kwargs={'rarhmm': torch.load(open(os.path.dirname(__file__)
                                          + '/envs/hybrid/models/poly_rarhmm_msd.pkl', 'rb'),
                                     map_location='cpu')}
    )
except:
    pass

try:
    register(
        id='HybridPendulum-ID-v0',
        entry_point='sds_numpy.envs:HybridPendulum',
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
        entry_point='sds_numpy.envs:HybridPendulumWithCartesianObservation',
        max_episode_steps=1000,
        kwargs={'rarhmm': torch.load(open(os.path.dirname(__file__)
                                          + '/envs/hybrid/models/neural_rarhmm_pendulum_cart.pkl', 'rb'),
                                     map_location='cpu')}
    )
except:
    pass
