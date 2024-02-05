import gymnasium as gym
import numpy as np
from gym_game.envs.agent2 import DoubleQAgent
import math
import torch
import matplotlib.pyplot as plt
import csv
import time
from pathlib import Path
import os


if __name__ == "__main__":
    choose_env = 3
    while choose_env!=1 and choose_env!=2:
        choose_env = int(input("Select the gym environment (1: cut or move, 2: move and cut)\n"))
    
    env = None
    env_str = None
    if choose_env == 1:
        env = gym.make("gym_game:gym_game/LawnMowingGame-v0", render_mode="human")
        env_str = "LawnMowingGame-v0"
    else:
        env = gym.make("gym_game:gym_game/LawnMowingGame-v1", render_mode="human")
        env_str = "LawnMowingGame-v1"

    print("Spazio delle osservazioni (griglia):",env.observation_space.shape)
    print("Spazio delle azioni: ",env.action_space.n)

    train_str = input("Do you want to start the train process? (True/False): ")
    train = train_str.lower() in ['true', '1', 'yes']

    gamma = 0.8
    batch = 128
    lr = 0.001
    reward_range = None
    if train:
        gamma = float(input("Insert the gamma value\n")) #0.8
        batch = int(input("Insert the batch size\n")) #128
        lr = float(input("Insert the learning rate value\n")) #0.001
        reward_range = str(input("Insert the reward range (stop penalty - min - max)\n"))
    else:
        print("Gamma: {}, Starting Epsilon: 1, Batch size: {}, Learning rate: {}".format(gamma, batch, lr))

    Path("models/model2").mkdir(parents=True, exist_ok=True)

    agent = DoubleQAgent(gamma=gamma, epsilon=1.0, epsilon_dec=0.995, lr=lr, mem_size=200000, batch_size=batch, epsilon_end=0.01, num_actions=env.action_space.n)
    csv_row = {"model": "model2",
                "environment": env_str,
                "gamma":gamma,
                "epsilon":1.0,
                "batch_size":batch,
                "lr":lr,
                "num_train_game":0,
                "best_train_game":0,
                "max_train_score":0,
                "num_train_actions":0,
                "train_model_actions":0,
                "num_test_game":0,
                "best_test_game":0,
                "max_test_score":0,
                "mean_test_score":0,
                "reward_range": reward_range
                }    
    
    if train:
        scores = []
        eps_history = []
        max_score = -math.inf
        best_game=0
        best_model_actions=0
        best_agent_steps=0
        games=list()
        LEARN_EVERY = 4

        n_games = int(input("Insert the number of train games\n"))
        csv_row["num_train_game"] = n_games

        print("*********************** TRAIN ***********************")
        for i in range(n_games):
            terminated = False
            truncated = False
            score = 0
            state = env.reset()[0]
            steps = 0

            while not (terminated or truncated):
                position = env.get_lawnmower_position()
                action = agent.choose_action(state, position)
                new_state, reward, terminated, truncated, _ = env.step(action)
                agent.save(state, action, reward, new_state, terminated)
                state = new_state
                if steps > 0 and steps % LEARN_EVERY == 0:
                    agent.learn()
                steps += 1
                score += reward

            if score > max_score and agent.epsilon<=0.3:
                best_model_actions=agent.model_actions
                best_game=i
                max_score=score
                best_agent_steps=steps
                configuration = agent.get_model_configuration()
                torch.save(configuration, "models/model2/ddqn_model_pytorch.pth")

            games.append(i)
            eps_history.append(agent.epsilon)
            scores.append(score)
            avg_score = np.mean(scores[max(0, i-100):(i+1)])

            print("Game: {}, Num. Actions: {}, Model actions:{} Score: {}, Average score: {}, Epsilon: {} \n".format(i, steps, agent.model_actions, score, avg_score, agent.epsilon))
            agent.model_actions=0
            
        print("Best game: {} Max score: {}, Agent actions: {}, Model actions {}, Avarage last 100 game: {}".format(best_game, max_score, best_agent_steps, best_model_actions, avg_score))

        #CSV train columns
        csv_row["best_train_game"] = best_game
        csv_row["max_train_score"] = max_score
        csv_row["num_train_actions"] = best_agent_steps
        csv_row["train_model_actions"] = best_model_actions

        #PLOT
        #Plot scores
        plt.figure(figsize=(11, 10)) #width, height
        plt.subplots_adjust(hspace=0.38)
        plt.suptitle('Train trend score')

        plt.subplot(2,1,1) #Nrow, Ncolumn, index
        plt.title('Trend Score')
        plt.plot(games, scores, color="blue")
        plt.xlabel('Games')
        plt.ylabel('Score')
        
        #Plot epsilon
        plt.subplot(2,1,2)
        plt.title('Trend Epsilon')
        plt.plot(games, eps_history, color="orange")
        plt.xlabel('Games')
        plt.ylabel('Epsilon')

        plt.savefig("models/model2/train_trend_model2.png")
        plt.show()
        time.sleep(3)
        plt.close()

    print("*********************** TEST ***********************")
    if train:
        agent.update_model(torch.load("models/model2/ddqn_model_pytorch.pth"))
    else:
        agent.update_model(torch.load("models/model2/best_ddqn_model_pytorch.pth"))

    num_games = int(input("Insert the number of games to test the agent:\n"))

    trend_test = list()
    games=list()
    observation, _ = env.reset()
    best_test_game=0
    best_test_score= -math.inf
    best_test_action=0
    test_steps=0

    for i in range(num_games):
        game_reward = 0
        done = False
        print("*********************** START GAME {} *********************** ".format(i))
        count_actions = 0
        while not done:
            test_steps+=1
            env.render()
            obs = np.array(observation) #Transform the observation in numpy array
            position = env.get_lawnmower_position()
            action = agent.choose_action(obs, position, test_mode=True)
            observation, reward, done, _, info = env.step(action)
            game_reward += reward
            count_actions += 1
            if not done and count_actions == 250:
               break
            
        
        print("Game: {}, Points: {}, Num. Actions: {}\n".format(i, game_reward,test_steps))
        
        games.append(i)
        trend_test.append(game_reward)

        if(game_reward>best_test_score):
            best_test_score = game_reward
            best_test_game = i
            best_test_actions = agent.model_actions

        test_steps=0
        agent.model_actions=0
        observation, _ = env.reset()

    print("Best game in test: {}, Score: {}, Num. actions: {}, Mean test score: {}".format(best_test_game, best_test_score, best_test_actions, np.mean(trend_test)))

    #CSV test columns
    csv_row["num_test_game"] = num_games
    csv_row["best_test_game"] = best_test_game
    csv_row["max_test_score"] = best_test_score
    csv_row["mean_test_score"] = np.mean(trend_test)

    #PLOT TEST
    plt.figure(figsize=(10, 5))
    plt.plot(games, trend_test, label='Score', color='blue')
    plt.title('Trend Score in TEST (2° approach)')
    plt.xlabel('Games')
    plt.ylabel('Points')
    plt.legend()
    plt.xticks(games)
    if train:
        plt.savefig("models/model2/test_trend_model2.png")
    plt.show()
    time.sleep(3)
    plt.close()

    env.close()

    #Save data in csv
    if train:
        if os.path.exists("models/model2/model2_configuration_and_performance.csv"):
            with open("models/model2/model2_configuration_and_performance.csv", "a", encoding="utf-8", newline="") as csvfile:
                fieldnames = list(csv_row.keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(csv_row)
        else:
            with open("models/model2/model2_configuration_and_performance.csv", "w", encoding="utf-8", newline="") as csvfile:
                fieldnames = list(csv_row.keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(csv_row)