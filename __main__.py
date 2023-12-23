import gymnasium as gym
import gym_game
import pygame
import time
from gym_game.envs.agent import MyAgent
import numpy as np
from gym_game.envs.model import DeepQNetwork
import torch
import os
import math


if __name__ == "__main__":
    env = gym.make("gym_game:gym_game/LawnMowingGame-v0", render_mode="human")
    print("Spazio delle osservazioni (griglia):",env.observation_space.shape)
    print("Spazio delle azioni: ",env.action_space.n)
    
    
    input_dim = (8, 8)
    observation, _ = env.reset()

    #m = DeepQNetwork(lr=0.003, input_dim=input_dim, n_actions=7)
    #print(m.fc1)


    agent = MyAgent(gamma=0.8, epsilon=1.0, batch_size=128, n_actions=7, eps_end=0.01, input_dim=(8,8), lr=0.001)
    env.render()
    scores, eps_history = [], []
    max_score = -math.inf

    print("*********************** TRAIN ***********************")
    n_games = 1000
    agent.set_model_on_train_mode()

    for i in range(n_games+10):
        print("Partita ",i)
        score = 0
        done = False
        #time.sleep(5)
        j = 0
        flag = True
        while not done:
            env.render()
            obs = np.array(observation) #Transform the observation in numpy array
            action = agent.choose_action(obs)
            
            new_observation, reward, done, _, info = env.step(action)
            score = score+reward

            agent.store_transition(observation, action, reward, new_observation, done)
            if i >=10:
                agent.learn()
            #if j==65:
            #    break
            observation = new_observation
            j = j+1
        
        scores.append(score)
        eps_history.append(agent.epsilon)
        
        avg_score = np.mean(scores[-100:]) #average of the last 100 scores

        print("Partita: {}, Num. Actions: {}, Score: {}, Average score: {}, Epsilon: {}".format(i, j, score, avg_score, agent.epsilon))

        if score>max_score and i>=10:
            max_score = score
            configuration = agent.get_model_configuration()
            torch.save(configuration, "dql_model_pytorch.pth")
        
        observation, _ = env.reset()


    print("Max score: {}, Avarage last 100 game: {}".format(max_score,np.mean(scores[-100:])))


    print("*********************** TEST ***********************")
    agent.update_model(torch.load("dql_model_pytorch.pth"))
    agent.set_model_on_test_mode()
    num_games = int(input("Insert the number of games to test the agent:\n"))

    trend_game = list()
    observation, _ = env.reset()

    for i in range(num_games):
        game_reward = 0
        done = False
        print("*********************** START GAME {} *********************** ".format(i))
        while not done:
            obs = np.array(observation) #Transform the observation in numpy array
            action = agent.choose_action(obs)
            observation, reward, done, _, info = env.step(action)
            game_reward = game_reward + reward
            env.render()
        
        print("Partita {}, points {}\n\n".format(i, game_reward))
        trend_game.append(game_reward)
        observation, _ = env.reset()
    
    print("Max score is: ",max(trend_game))

    env.close()