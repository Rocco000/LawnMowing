import gymnasium as gym
from gym_game.envs.agent import MyAgent
import numpy as np
import torch
import os
import math
import matplotlib.pyplot as plt


if __name__ == "__main__":
    env = gym.make("gym_game:gym_game/LawnMowingGame-v0", render_mode="human")
    print("Spazio delle osservazioni (griglia):",env.observation_space.shape)
    print("Spazio delle azioni: ",env.action_space.n)

    agent = MyAgent(gamma=0.8, epsilon=1.0, batch_size=128, n_actions=7, eps_end=0.01, input_dim=(8,8), lr=0.001)

    train_str = input("Do you want to start the train process? (True/False): ")
    train = train_str.lower() in ['true', '1', 'yes']
    if train:
        print("*********************** TRAIN ***********************")
        observation, _ = env.reset()
        scores, eps_history = [], []
        max_score = -math.inf
        LEARN_EVERY = 4
        n_games = 100
        agent.set_model_on_train_mode()
        best_game = 0
        best_model_actions = 0
        games = list()

        #PLOT
        plt.ion()  # Active the Matplotlib interactive mode

        #Plot score
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.set_title('Trend Score')
        ax1.set_xlabel('Games')
        ax1.set_ylabel('Score')
        ax1.legend()
        
        #Plot epsilon
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.set_title('Trend Epsilon')
        ax2.set_xlabel('Games')
        ax2.set_ylabel('Epsilon')
        ax2.legend()

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
                best_model_actions = agent.model_actions
                configuration = agent.get_model_configuration()
                torch.save(configuration, "dql_model_pytorch.pth")
            
            observation, _ = env.reset()
            agent.model_actions = 0

            #UPDATE PLOTS
            ax1.plot(games, scores, label='Score', color='blue')
            ax1.grid(True)
            ax2.plot(games, eps_history, label='Epsilon', color='orange')
            ax2.grid(True)

            fig1.canvas.draw()
            fig1.canvas.flush_events()
            fig2.canvas.draw()
            fig2.canvas.flush_events()

            plt.pause(0.5)


        print("Best game: {}, Max score: {}, Model actions: {}, Avarage last 100 games: {}".format(best_game, max_score, best_model_actions, np.mean(scores[-100:])))

        plt.ioff()  # Deactive the Matplotlib interactive mode
        plt.show()

    print("*********************** TEST ***********************")
    agent.update_model(torch.load("dql_model_pytorch.pth"))
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
    plt.figure(figsize=(10, 5))
    plt.plot(games, trend_test, label='Score', color='blue')
    plt.title('Trend Score in TEST (1° approach)')
    plt.xlabel('Games')
    plt.ylabel('Points')
    plt.legend()
    plt.show()

    env.close()