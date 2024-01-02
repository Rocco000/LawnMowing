import gymnasium as gym
import numpy as np
import math
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
import csv
import time


if __name__ == "__main__":
    env = gym.make("gym_game:gym_game/LawnMowingGame-v0", render_mode="human")
    print("Spazio delle osservazioni (griglia):",env.observation_space.shape)
    print("Spazio delle azioni: ",env.action_space.n)

    g = float(input("Insert the gamma value\n")) #0.8
    b = int(input("Insert the batch size\n")) #128
    l = float(input("Insert the learning rate value\n")) #0.001
    reward_range = str(input("Insert the reward range (min-max)\n")) 

    csv_row = {"model": "model3",
                "environment": "LawnMowingGame-v0",
                "gamma":g,
                "epsilon":1.0,
                "batch_size":b,
                "lr":l,
                "num_train_steps":0,
                "num_test_game":0,
                "best_test_game":0,
                "max_test_score":0,
                "mean_test_score":0,
                "reward_range": reward_range
            }
    
    model = DQN(policy= "MlpPolicy", #fully connected network
                env= env,
                learning_rate=l,
                buffer_size= 15000, #to store the old transaction made by th agent
                learning_starts= 200, #after 100 model steps, start the learning step. This step will be stored
                batch_size= b,
                gamma= g,
                train_freq= 1, #after 1 episode it will update the model
                gradient_steps= 1,
                target_update_interval= 50, #after 20 steps, the target model will be updated
                verbose=1
                )
    
    train_str = input("Do you want to start the train process? (True/False): ")
    train = train_str.lower() in ['true', '1', 'yes']
    if train:
        print("*********************** TRAIN ***********************")
        n_steps = int(input("Insert the number of train steps\n")) #100000
        csv_row["num_train_steps"] = n_steps
        model.learn(total_timesteps=n_steps)
        model.save("models/model3/dql_model_baseline3")

    print("*********************** TEST ***********************")
    num_games = int(input("Insert the number of games to test the agent:\n"))
    model.load("models/model3/dql_model_baseline3", env=env)

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

    #CSV test columns
    csv_row["num_test_game"] = num_games
    csv_row["best_test_game"] = best_test_game
    csv_row["max_test_score"] = best_test_score
    csv_row["mean_test_score"] = np.mean(trend_test)

    #PLOT TEST
    plt.figure(figsize=(10, 5))
    plt.plot(games, trend_test, label='Score', color='blue')
    plt.title('Trend Score in TEST (baseline)')
    plt.xlabel('Games')
    plt.ylabel('Points')
    plt.legend()
    plt.savefig("models/model3/test_trend_model3.png")
    plt.show()
    time.sleep(3)
    plt.close()

    env.close()

    #Save data in csv
    with open("models/model3/model3_configuration_and_performance.csv","a") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_row.values())


    