import gymnasium as gym
import numpy as np
import math
import matplotlib.pyplot as plt
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
    
    train_str = input("Do you want to start the train process? (True/False): ")
    train = train_str.lower() in ['true', '1', 'yes']
    if train:
        print("*********************** TRAIN ***********************")
        model.learn(total_timesteps=100000)
        model.save("dql_model_baseline3")

    print("*********************** TEST ***********************")
    num_games = int(input("Insert the number of games to test the agent:\n"))
    model.load("dql_model_baseline3", env=env)

    model_env = model.get_env()
    observation = model_env.reset()

    trend_test = list()
    games = list()
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

        trend_test.append(game_reward)
        model_env.reset()

        if game_reward > best_test_score:
            best_test_score = game_reward
            best_test_game = i
            best_test_action = test_steps
        
        test_steps = 0
        games.append(i)

    
    print("Best game in test: {}, Score: {}, Num. actions: {}".format(best_test_game, best_test_score, best_test_action))
    plt.figure(figsize=(10, 5))
    plt.plot(games, trend_test, label='Score', color='blue')
    plt.title('Trend Score in TEST (baseline)')
    plt.xlabel('Games')
    plt.ylabel('Points')
    plt.legend()
    plt.show()

    env.close()


    