import gymnasium as gym
import numpy as np
from gym_game.envs.agent2 import DoubleQAgent
import math
import torch
import matplotlib.pyplot as plt
import csv
import time



env = gym.make("gym_game:gym_game/LawnMowingGame-v1", render_mode="human")
print("Spazio delle osservazioni (griglia):",env.observation_space.shape)
print("Spazio delle azioni: ",env.action_space.n)

gamma = float(input("Insert the gamma value\n")) #0.8
batch = int(input("Insert the batch size\n")) #128
lr = float(input("Insert the learning rate value\n")) #0.001
reward_range = str(input("Insert the reward range (min-max)\n"))

agent = DoubleQAgent(gamma=gamma, epsilon=1.0, epsilon_dec=0.995, lr=lr, mem_size=200000, batch_size=batch, epsilon_end=0.01)
csv_row = {"model": "model1",
            "environment": "LawnMowingGame-v0",
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
train_str = input("Do you want to start the train process? (True/False): ")
train = train_str.lower() in ['true', '1', 'yes']
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
agent.update_model(torch.load("models/model2/ddqn_model_pytorch.pth"))

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
    while not done:
        test_steps+=1
        env.render()
        obs = np.array(observation) #Transform the observation in numpy array
        action = agent.choose_action(obs, test_mode=True)
        observation, reward, done, _, info = env.step(action)
        game_reward += reward
        
    
    print("Game: {}, Points: {}, Num. Actions: {}, Mean test score: {}\n\n".format(i, game_reward,test_steps,np.mean(trend_test)))
    
    games.append(i)
    trend_test.append(game_reward)

    if(game_reward>best_test_score):
        best_test_score = game_reward
        best_test_game = i
        best_test_actions = agent.model_actions

    test_steps=0
    agent.model_actions=0
    observation, _ = env.reset()

print("Best game in test: {}, Score: {}, Num. actions: {}".format(best_test_game, best_test_score, best_test_actions))

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
plt.savefig("models/model2/test_trend_model2.png")
plt.show()
time.sleep(3)
plt.close()

env.close()

#Save data in csv
with open("models/model2/model2_configuration_and_performance.csv","a") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(csv_row.values())