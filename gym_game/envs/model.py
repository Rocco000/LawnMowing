import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DeepQNetwork(nn.Module):
    #input_dim is the size of the game grid
    def __init__(self, lr, input_dim, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dim = input_dim[0]*input_dim[0]
        self.n_actions = n_actions
        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = state.view(-1,64) #Transform the game grid in a unidimensional array
        #print("Dimensione input flattened: ",x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x