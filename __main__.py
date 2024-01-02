import gymnasium as gym
from gym_game.envs.agent import MyAgent
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
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

    agent = MyAgent(gamma=g, epsilon=1.0, batch_size=b, n_actions=7, eps_end=0.01, input_dim=(8,8), lr=l)
    csv_row = {"model": "model1",
                "environment": "LawnMowingGame-v0",
                "gamma":g,
                "epsilon":1.0,
                "batch_size":b,
                "lr":l,
                "num_train_game":1000,
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
    train_str = input("Do you want to start the train process? (True/False): ")
    train = train_str.lower() in ['true', '1', 'yes']
    if train:
        print("*********************** TRAIN ***********************")
        observation, _ = env.reset()
        scores, eps_history = [], []
        max_score = -math.inf
        LEARN_EVERY = 4
        n_games = int(input("Insert the number of train games\n"))
        csv_row["num_train_game"] = n_games
        agent.set_model_on_train_mode()
        best_game = 0
        best_model_actions = 0
        best_agent_steps = 0
        games = list()

        for i in range(n_games):
            score = 0
            done = False
            steps = 0
            flag = True
            while not done:
                env.render()
                obs = np.array(observation) #Transform the observation in numpy array
                action = agent.choose_action(obs)
                
                new_observation, reward, done, _, info = env.step(action)
                score = score+reward

                agent.store_transition(observation, action, reward, new_observation, done)
                if steps>0 and steps % LEARN_EVERY == 0:
                    agent.learn()
                
                observation = new_observation
                steps = steps+1
            
            games.append(i)
            scores.append(score)
            eps_history.append(agent.epsilon)
            
            avg_score = np.mean(scores[-100:]) #average of the last 100 scores

            print("Game: {}, Num. Actions: {}, Num. Model Actions: {}, Score: {}, Average score: {}, Epsilon: {}".format(i, steps, agent.model_actions, score, avg_score, agent.epsilon))

            if score>max_score:
                max_score = score
                best_game = i
                best_agent_steps = steps
                best_model_actions = agent.model_actions
                configuration = agent.get_model_configuration()
                torch.save(configuration, "models/model1/dql_model_pytorch.pth")
            
            observation, _ = env.reset()
            agent.model_actions = 0

        print("Best game: {}, Max score: {}, Agent actions: {}, Model actions: {}, Avarage last 100 games: {}".format(best_game, max_score, best_agent_steps, best_model_actions, np.mean(scores[-100:])))
        
        #CSV train columns
        csv_row["best_train_game"] = best_game
        csv_row["max_train_score"] = max_score
        csv_row["num_train_actions"] = best_agent_steps
        csv_row["train_model_actions"] = best_model_actions

        #PLOT
        #Plot score
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.set_title('Trend Score')
        ax1.set_xlabel('Games')
        ax1.set_ylabel('Score')
        ax1.plot(games, scores, label='Score', color='blue')
        ax1.legend()
        
        #Plot epsilon
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.set_title('Trend Epsilon')
        ax2.set_xlabel('Games')
        ax2.set_ylabel('Epsilon')
        ax2.plot(games, eps_history, label='Epsilon', color='orange')
        ax2.legend()

        plt.show()
        time.sleep(3)
        plt.close()


    print("*********************** TEST ***********************")
    agent.update_model(torch.load("models/model1/dql_model_pytorch.pth"))
    agent.set_model_on_test_mode()
    num_games = int(input("Insert the number of games to test the agent:\n"))

    trend_test = list()
    games = list()
    best_test_game = 0
    best_test_score = -math.inf
    best_test_action = 0
    test_steps = 0

    observation, _ = env.reset()

    for i in range(num_games):
        game_reward = 0
        done = False
        print("*********************** START GAME {} *********************** ".format(i))
        while not done:
            env.render()
            obs = np.array(observation) #Transform the observation in numpy array
            action = agent.choose_action(obs, test_mode=True)
            observation, reward, done, _, info = env.step(action)
            game_reward = game_reward + reward
            test_steps = test_steps + 1
        
        print("Game: {}, Points: {}, Num. actions: {}\n\n".format(i, game_reward, test_steps))
        trend_test.append(game_reward)
        observation, _ = env.reset()

        if game_reward>best_test_score:
            best_test_score = game_reward
            best_test_game = i
            best_test_action = agent.model_actions
        
        agent.model_actions = 0
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
    plt.title('Trend Score in TEST (1° approach)')
    plt.xlabel('Games')
    plt.ylabel('Points')
    plt.legend()
    plt.savefig("models/model1/test_trend_model1.png")
    plt.show()
    time.sleep(3)
    plt.close()

    env.close()

    #Save data in csv
    with open("models/model1/model1_configuration_and_performance.csv","a") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_row.values())