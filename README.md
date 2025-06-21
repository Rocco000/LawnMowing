# LawnMowing
  <p align="center">
    <img src="https://github.com/Rocco000/LawnMowing/blob/main/environment_rendering.png" alt="LawnMowing Environment" width="300">
  </p>

  This repository contains a Reinforcement Learning (RL) implementation designed to train agents that autonomously cut grass in a dynamically changing environment. The game simulates real-world complexity including weather conditions, obstacles, and varying grass height.

## 🎮 Domain Description
The domain simulates a **2D lawn mowing environment**, rendered with PyGame, where an autonomous agent navigates an 8x8 grid to cut grass. The environment includes:
- **Grass** at four levels: low, medium, high, and already cut
- **Obstacles**: rocks and trees
- **Weather**: sunny, cloudy, and rainy — which influences the rate of grass regrowth
- The **agent** is represented by a lawnmower avatar and must decide how to move and how intensely to cut.

The goal of the agent is to **maximize rewards** by efficiently mowing the lawn, avoiding penalties, and adapting to environmental changes.

## 🌍 Environment Description
The environment is custom-built using Gymnasium and includes several unique mechanics:
  ### Configuration
  We developed two environment configurations to evaluate the model performance:
  - **Environment 1**: Movement and mowing are separate actions.
  - **Environment 2**: Movement includes mowing.

To thoroughly assess the effect of the environment design on agent learning, we defined **three different reward-penalty configurations** for each type. These variations helped us identify which configurations yielded the most effective and efficient agent behavior.

  ### 🌱 Grass Mechanism
  Grass Regrowth is probabilistic and driven by weather:
  - Sunny ☀️: grows **3 times** before weather changes
  - Cloudy ☁️: grows **1 time**
  - Rainy 🌧️: grows **2 times**
  
  ### ⚠️ Obstacles
  - Rocks and trees impose cutting penalties.
  - Invalid actions, such as trying to move outside the grid, result in significant penalties.
  
  ### 💾 State encoding
  - Each cell holds a value indicating grass height or obstacle type (0–5, where 0-3 represents the grass height, 4 tree, and 5 rock).
  - The **agent's position** is encoded by adding **+6** to the base cell value.
  
  Total state space:
  - **Environment 1**: $64^7$ unique states
  - **Environment 2**: $64^{12}$ unique states

  This high-dimensional space necessitates the use of **Deep Q-Learning**.

## 🤖 RL Algorithms used
Three different RL agents were implemented:
1. **Deep Q-Learning Agent (DQN)**:
   - Uses a single neural network
   - Learns Q-values from transitions using experience replay
2. **Double Deep Q-Learning Agent (DDQN)**:
   - Utilizes two neural networks to stabilize learning
3. **Stable-Baselines3 DQN Agent**:
   - Uses the _MlpPolicy_ from the [stable_baselines3](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html) library

  ### ⚙️ Hyperparameters
  - &gamma; (Gamma): 0.8
  - &epsilon; (Epsilon): decays from 1 to 0.01
  - &alpha; (Learning rate): 0.001

## 💻 Project environment setup
  To replicate this project, follow this steps:
  1. Clone the repository by running:
  ```bash
  https://github.com/Rocco000/LawnMowing.git
  ```
  2. Make sure to install the required dependencies by running:
  ```bash
  pip install -r requirements.txt
  ```

  3. Train an agent:
  ```bash
  __main__.py # Train DQN
  __mainDDQN__.py # Train DDQN
  __main_baseline__.py # Train stable_baseline3 model
  ```

## 📊 Results
All three agents were trained and tested across multiple environment configurations. The results were measured in terms of cumulative rewards:
| Agent | Environment | Max Train Score | Max Test Score |
| :---: | :---:       | :---:           | :---:          |
| DQN   | 2-1         | 70              | 49             |
| DDQN  | 2-3         | 200             | 80             |
| Stable-baselines3 | 2-1 | 38.4 (avg on last 100 steps) | 208 | 

The **stable-baselines3 agent** achieved the **highest test performance** but underperformed when reloaded without retraining.

## 🛠️ Future works
- [ ] ⌛ **Penalty Over Time**: Introduce time-based penalties to encourage faster mowing.
- [ ] ⚙️ **Reward Shaping**: Fine-tune reward schemes to improve sample efficiency

## 👨🏻‍💻 Contributors
<a href="https://github.com/Rocco000/LawnMowing/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Rocco000/LawnMowing" />
</a>
  
