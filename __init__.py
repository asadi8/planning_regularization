# vim: tabstop=2 softtabstop=2 shiftwidth=2 expandtab
from gym.envs.registration import registry, register, make, spec

register(
    id='ModelBasedAtariFreeway-v0',
    entry_point='gym_model_atari.model:ModelBasedAtariEnv',
    kwargs={
      'name':'freeway'
    }
)
