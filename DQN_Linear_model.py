### modified from: https://github.com/patrickloeber/snake-ai-pytorch

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("GPU is available:", torch.cuda.is_available())


class Linear_QNet(nn.Module):
    def __init__(self, input_dim):
        super(Linear_QNet, self).__init__()
        self.linear1 = nn.Linear(input_dim, 256)
        self.linear2 = nn.Linear(256, 120)
        self.linear3 = nn.Linear(120, 3) 
        self.to(device)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = self.linear3(x)
        return x


    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = torch.nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        pred = self.model(state)
        target = pred.clone().to(device)

        with torch.no_grad():
            Q_new = reward + self.gamma * torch.max(self.model(next_state)) if not done else reward
        
        target[0][action] = Q_new
        loss = self.criterion(target, pred)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
