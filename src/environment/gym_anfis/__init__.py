from gym.envs.registration import register

register(
    id='anfis-v0',
    entry_point='gym_anfis.envs:Environment',
)
