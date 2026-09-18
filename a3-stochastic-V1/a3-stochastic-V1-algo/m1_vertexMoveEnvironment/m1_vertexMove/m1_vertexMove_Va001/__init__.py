from gym.envs.registration import register

# register(
#     id='vertexMove-V1',
#     entry_point='vertexMove.envs:gym_vertexMoveV1',
#     max_episode_steps=100,
# )

# register(
#     id='vertexMove-V002',
#     entry_point='gym.envs.vertexMove.envs:gym_vertexMoveV1',
#     max_episode_steps=100,
# )

# register(
#     id='m1_vertexMove_Va002',
#     entry_point='m1_vertexMove_Va001.envs:m1_vertexMove_Va001',
#     max_episode_steps=int(1e7),#100  103
# )


register(
    id='m1_vertexMove_Va003',
    entry_point='m1_vertexMove_Va001.envs:m1_vertexMove_Va001',
    max_episode_steps=int(1e7),#100  103
)