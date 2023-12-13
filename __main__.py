import gymnasium as gym
import gym_game
import pygame
import time
from gym_game.envs.agent import MyAgent
import numpy as np
from gym_game.envs.model import DeepQNetwork
import torch as T
import torch.nn.functional as F


if __name__ == "__main__":
    env = gym.make("gym_game:gym_game/LawnMowingGame-v0", render_mode="human")
    print("Spazio delle osservazioni (griglia):",env.observation_space["grid"].shape)
    print("Spazio delle osservazioni (agente):",env.observation_space["agent"].shape)
    print("Spazio delle azioni: ",env.action_space.n)
    
    
    input_dim = (8, 8)
    observation, _ = env.reset()

    m = DeepQNetwork(lr=0.003, input_dim=input_dim, n_actions=7)
    print(m.fc1)


    agent = MyAgent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=7, eps_end=0.01, input_dim=(8,8), lr=0.001)
    env.render()
    scores, eps_history = [], []

    n_games = 1000

    for i in range(n_games):
        score = 0
        done = False
        observation, _ = env.reset()
        j = 0
        flag = True
        while not done:
            env.render()
            obs = np.array(observation["grid"]) #Transform the observation in numpy array
            action = agent.choose_action(obs)
            print("azione intrapresa: ",action)
            new_observation, reward, done, _, info = env.step(action)
            score += reward

            #Start the thread to update weather and grass
            if not done and flag:
                flag = False
                env.create_weather_thread()
                

            agent.store_transition(observation["grid"], action, reward, new_observation["grid"], done)

            agent.learn()
            if j==65:
                break
            observation = new_observation
            j = j+1
        
        scores.append(score)
        eps_history.append(agent.epsilon)
        
        avg_score = np.mean(scores[-100:]) #average of the last 100 scores

        print("Partita: {}, Score: {}, Average score: {}, Epsilon: {}".format(i, score, avg_score, agent.epsilon))

    print("Max score: ",max(scores))
    
    env.close()