import gymnasium as gym
import numpy as np
from gym_game.envs.agent2 import DoubleQAgent
import math
import torch

LEARN_EVERY = 4
n_games=1000
env = gym.make("gym_game:gym_game/LawnMowingGame-v1", render_mode="human")
print("Spazio delle osservazioni (griglia):",env.observation_space.shape)
print("Spazio delle azioni: ",env.action_space.n)

agent = DoubleQAgent(gamma=0.99, epsilon=1.0, epsilon_dec=0.995, lr=0.001, mem_size=200000, batch_size=128, epsilon_end=0.01)
    
scores = []
eps_history = []
max_score = -math.inf
best_game=0
best_model_actions=0

print("*********************** TRAIN ***********************")
for i in range(n_games):
    terminated = False
    truncated = False
    score = 0
    state = env.reset()[0]
    steps = 0
    while not (terminated or truncated):
        action = agent.choose_action(state)
        new_state, reward, terminated, truncated, _ = env.step(action)
        agent.save(state, action, reward, new_state, terminated)
        state = new_state
        if steps > 0 and steps % LEARN_EVERY == 0:
            agent.learn()
        steps += 1
        score += reward

    if score > max_score:
        best_model_actions=agent.model_actions
        best_game=i
        max_score=score
        configuration = agent.get_model_configuration()
        torch.save(configuration, "ddqn_model_pytorch.pth")
    eps_history.append(agent.epsilon)
    scores.append(score)
    avg_score = np.mean(scores[max(0, i-100):(i+1)])

    print("Game: {}, Num. Actions: {}, Model actions:{} Score: {}, Average score: {}, Epsilon: {} \n".format(i, steps, agent.model_actions, score, avg_score, agent.epsilon))
    agent.model_actions=0
    

print("Best game: {} Max score: {}, Model actions: {} Avarage last 100 game: {}".format(best_game,best_model_actions,max_score,avg_score))

print("*********************** TEST ***********************")
agent.update_model(torch.load("ddqn_model_pytorch.pth"))

num_games = int(input("Insert the number of games to test the agent:\n"))

trend_game = list()
observation, _ = env.reset()
best_test_game=0
best_test_score= -math.inf
best_test_model_actions=0

for i in range(num_games):
    game_reward = 0
    done = False
    test_steps=0
    print("*********************** START GAME {} *********************** ".format(i))
    while not done:
        test_steps+=1
        env.render()
        obs = np.array(observation) #Transform the observation in numpy array
        action = agent.choose_action(obs, test_mode=True)
        observation, reward, done, _, info = env.step(action)
        game_reward = game_reward + reward
        
    
    print("Game: {}, Points: {}, Num. Actions {}, Model Actions: {}\n\n".format(i, game_reward,test_steps,agent.model_actions))
    trend_game.append(game_reward)
    if(game_reward>best_test_score):
        best_test_score = game_reward
        best_test_game = i
        best_test_model_actions = agent.model_actions

    test_steps=0
    agent.model_actions=0
    observation, _ = env.reset()

print("Best game: {}, Model Actions: {}, Max Score: {}".format(best_test_game, best_test_model_actions, best_test_score))
env.close()