from gymnasium.envs.registration import register

#To register our environment
#id to identify our environment
#entry_point the environment class

register(
    id="gym_game/LawnMowingGame-v0",
    entry_point="gym_game.envs:LawnMowingEnvironment",
    max_episode_steps=2000,
)