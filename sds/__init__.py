import os
import torch

from gym.envs.registration import register

register(
    id='BouncingBall-ID-v0',
    entry_point='sds.envs:BouncingBall',
    max_episode_steps=1000,
)

register(
    id='Pole-ID-v0',
    entry_point='sds.envs:PoleWithWall',
    max_episode_steps=1000,
)

try:
    register(
        id='HybridPole-ID-v0',
        entry_point='sds.envs:HybridPoleWithWall',
        max_episode_steps=1000,
        kwargs={'rarhmm': torch.load(open(os.path.dirname(__file__)
                                          + '/envs/hybrid/models/rarhmm_pole.pkl', 'rb'),
                                     map_location='cpu')}
    )
except :
    pass

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

try:
    register(
        id='HybridPendulum-ID-v0',
        entry_point='sds.envs:HybridPendulum',
        max_episode_steps=1000,
        kwargs={'rarhmm': torch.load(open(os.path.dirname(__file__)
                                          + '/envs/hybrid/models/rarhmm_pendulum_polar.pkl', 'rb'),
                                     map_location='cpu')}
    )
except :
    pass

try:
    register(
        id='HybridPendulum-ID-v1',
        entry_point='sds.envs:HybridPendulumWithCartesianObservation',
        max_episode_steps=1000,
        kwargs={'rarhmm': torch.load(open(os.path.dirname(__file__)
                                          + '/envs/hybrid/models/rarhmm_pendulum_cart.pkl', 'rb'),
                                     map_location='cpu')}
    )
except:
    pass

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
