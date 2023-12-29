import gymnasium as gym
import numpy as np
import math
from stable_baselines3 import DQN


if __name__ == "__main__":
    env = gym.make("gym_game:gym_game/LawnMowingGame-v0", render_mode="human")
    print("Spazio delle osservazioni (griglia):",env.observation_space.shape)
    print("Spazio delle azioni: ",env.action_space.n)

    model = DQN(policy= "MlpPolicy", #fully connected network
                env= env,
                learning_rate=0.001,
                buffer_size= 15000, #to store the old transaction made by th agent
                learning_starts= 200, #after 100 model steps, start the learning step. This step will be stored
                batch_size= 128,
                gamma= 0.8,
                train_freq= 1,
                gradient_steps= 1,
                target_update_interval= 50, #after 20 steps, the target model will be updated
                verbose=1
                )

    print("*********************** TRAIN ***********************")
    model.learn(total_timesteps=100000)
    model.save("dql_model_baseline3")

    print("*********************** TEST ***********************")
    num_games = int(input("Insert the number of games to test the agent:\n"))

    model_env = model.get_env()
    observation = model_env.reset()

    trend_game = list()
    best_test_game = 0
    best_test_score = -math.inf
    best_test_action = 0
    test_steps = 0

    for i in range(num_games):
        game_reward = 0
        done = False
        print("*********************** START GAME {} *********************** ".format(i))
        while not done:
            model_env.render()
            action, _state = model.predict(observation, deterministic=True)
            observation, reward, done, info = model_env.step(action)
            game_reward = game_reward + reward
            test_steps = test_steps + 1
        
        print("Game: {}, Points: {}\n\n".format(i, game_reward))

        trend_game.append(game_reward)
        model_env.reset()

        if game_reward > best_test_score:
            best_test_score = game_reward
            best_test_game = i
            best_test_action = test_steps
        
        test_steps = 0

    
    print("Best game in test: {}, Score: {}, Num. actions: {}".format(best_test_game, best_test_score, best_test_action))
    env.close()


    