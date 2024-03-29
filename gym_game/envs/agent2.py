import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import collections # For dequeue for the memory buffer
import random


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MemoryBuffer(object):
    def __init__(self, max_size):
        self.memory_size = max_size
        self.trans_counter=0 # num of transitions in the memory
                             # this count is required to delay learning
                             # until the buffer is sensibly full
        self.index=0         # current pointer in the buffer
        self.buffer = collections.deque(maxlen=self.memory_size)
        self.transition = collections.namedtuple("Transition", field_names=["state", "action", "reward", "new_state", "terminal"])

    
    def save(self, state, action, reward, new_state, terminal):
        t = self.transition(state, action, reward, new_state, terminal)
        self.buffer.append(t)
        self.trans_counter = (self.trans_counter + 1) % self.memory_size

    def random_sample(self, batch_size):
        assert len(self.buffer) >= batch_size # should begin sampling only when sufficiently full
        transitions = random.sample(self.buffer, k=batch_size) # number of transitions to sample
        states = torch.stack([torch.from_numpy(e.state) for e in transitions if e is not None]).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in transitions if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in transitions if e is not None])).float().to(device)
        new_states = torch.stack([torch.from_numpy(e.new_state) for e in transitions if e is not None]).float().to(device)
        terminals = torch.from_numpy(np.vstack([e.terminal for e in transitions if e is not None]).astype(np.uint8)).float().to(device)
        return states, actions, rewards, new_states, terminals

class QNN(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(QNN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        
    def forward(self, state):
        x = state.view(-1,64)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)

class Agent(object):
    def __init__(self, gamma=0.99, epsilon=1.0, batch_size=128, lr=0.001,
                 epsilon_dec=0.996,  epsilon_end=0.01,
                 mem_size=1000000, num_actions:int=7):
        self.gamma = gamma # alpha = learn rate, gamma = discount
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec # decrement of epsilon for larger spaces
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.memory = MemoryBuffer(mem_size)
        self.model_actions=0
        self.num_actions = num_actions

    def save(self, state, action, reward, new_state, done):
        self.memory.save(state, action, reward, new_state, done)

    def choose_action(self, state, agent_position, test_mode=False):
        rand = random.uniform(0,1)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        action = None

        if rand > self.epsilon or test_mode: 
            self.q_func.eval()
            
            with torch.no_grad():
                valid = False
                count = 0
                while not valid:
                    action_values = self.q_func(state)
                    action = np.argmax(action_values.cpu().data.numpy())
                        
                    valid = self.check_action(action, agent_position)
                    count += 1

                    if count == 10 and not valid:
                        count += 1
                        while not valid:
                            action = np.random.choice([i for i in range(self.num_actions)])
                            valid = self.check_action(action, agent_position)


            if not test_mode:
                self.q_func.train()

            if count != 11:
                self.model_actions+=1
        else:
            # exploring: return a random action
            valid = False
            while not valid:
                action = np.random.choice([i for i in range(self.num_actions)])
                valid = self.check_action(action, agent_position)
        
        return action     
    
    def check_action(self, action, position):
        if position[0] == 0:
            if position[1] == 0 and action == 3:
                return False
            elif position[1] == 7 and action == 1:
                return False
            elif action == 2:
                return False
        elif position[0] == 7:
            if position[1] == 0 and action==3:
                return False
            elif position[1] == 7 and action==1:
                return False
            elif action == 0:
                return False
        elif position[1] == 0:
            if action == 3: #indietro
                return False
        elif position[1] == 7:
            if action == 1:
                return False
        
        return True

    def reduce_epsilon(self):
        self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min  
        
    def learn(self):
        raise Exception("Not implemented")
        
    def save_model(self, path):
        torch.save(self.q_func.state_dict(), path)

    def load_saved_model(self, path):
        self.q_func = QNN(64, self.num_actions, 42).to(device)
        self.q_func.load_state_dict(torch.load(path))
        self.q_func.eval()

    def update_model(self, configuration):
        self.q_func.load_state_dict(configuration)
        
    def get_model_configuration(self):
        return self.q_func.state_dict()
    
class DoubleQAgent(Agent):
    def __init__(self, gamma=0.99, epsilon=1.0, batch_size=128, lr=0.001,
                 epsilon_dec=0.996,  epsilon_end=0.01,
                 mem_size=1000000, replace_q_target = 100, num_actions:int=7):
        
        super().__init__(lr=lr, gamma=gamma, epsilon=epsilon, batch_size=batch_size,
             epsilon_dec=epsilon_dec,  epsilon_end=epsilon_end,
             mem_size=mem_size, num_actions=num_actions)

        self.replace_q_target = replace_q_target
        self.q_func = QNN(64, self.num_actions, 42).to(device)
        self.q_func_target = QNN(64, self.num_actions, 42).to(device)
        self.optimizer = optim.Adam(self.q_func.parameters(), lr=lr)
        
        
    def learn(self):
        if self.memory.trans_counter < self.batch_size: # wait before you start learning
            return
            
        # 1. Choose a sample from past transitions:
        states, actions, rewards, new_states, terminals = self.memory.random_sample(self.batch_size)
        
        # 2. Update the target values
        q_next = self.q_func_target(new_states).detach().max(1)[0].unsqueeze(1)
        q_updated = rewards + self.gamma * q_next * (1 - terminals)
        q = self.q_func(states).gather(1, actions)
        
        # 3. Update the main NN
        loss = F.mse_loss(q, q_updated)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 4. Update the target NN (every N-th step)
        if self.memory.trans_counter % self.replace_q_target == 0: # wait before you start learning
            for target_param, local_param in zip(self.q_func_target.parameters(), self.q_func.parameters()):
                target_param.data.copy_(local_param.data)
        # 5. Reduce the exploration rate
        self.reduce_epsilon()



    def save_model(self, path):
        super().save_model(path)
        torch.save(self.q_func.state_dict(), path+'.target')


    def load_saved_model(self, path):
        super().load_saved_model(path)
        self.q_func_target = QNN(64, self.num_actions, 42).to(device)
        self.q_func_target.load_state_dict(torch.load(path+'.target'))
        self.q_func_target.eval()