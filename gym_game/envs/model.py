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
        self.batch1 = nn.BatchNorm1d(128)

        self.fc2 = nn.Linear(128, 128)
        self.batch2 = nn.BatchNorm1d(128)
            
        self.fc3 = nn.Linear(128, 64)
        self.batch3 = nn.BatchNorm1d(64)
            
        self.fc4 = nn.Linear(64, 64)
        self.batch4 = nn.BatchNorm1d(64)
            
        self.fc5 = nn.Linear(64,32)
        self.batch5 = nn.BatchNorm1d(32) 

        self.fc6 = nn.Linear(32, n_actions)
        
        self.batch6 = nn.BatchNorm1d(n_actions)

        self.my_activation = nn.LeakyReLU(0.1)
        self.my_drop = nn.Dropout1d(p=0.2)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = state.view(-1,64) #Transform the game grid in a unidimensional array
        
        if x.shape[0] == 1:
            x = self.my_activation(self.fc1(x))
            x = self.my_activation(self.fc2(x))
            x = self.my_activation(self.fc3(x))
            x = self.my_activation(self.fc4(x))
            x = self.my_activation(self.fc5(x))
            x = self.fc6(x)
        else:
            x = self.my_activation(self.batch1(self.fc1(x)))
            x = self.my_activation(self.batch2(self.fc2(x)))
            x = self.my_drop(self.my_activation(self.batch3(self.fc3(x))))
            x = self.my_activation(self.batch4(self.fc4(x)))
            x = self.my_drop(self.my_activation(self.batch5(self.fc5(x))))
            x = self.batch6(self.fc6(x))
   
        return x