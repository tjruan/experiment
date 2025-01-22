import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import torch.nn.functional as F
import torch.nn.init as init
device = torch.device('cpu')
print("Device set to : cpu")
print("===============================================")

# set seed
np.random.seed(0)
torch.manual_seed(0)

class DNet(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super(DNet, self).__init__()
        self.dnet = nn.Sequential(
                        nn.Linear(state_dim, hidden_dim),
                        nn.ELU(),
                        nn.Linear(hidden_dim, hidden_dim), 
                        nn.ELU(),
                        nn.Linear(hidden_dim, 1)
                    )

    def forward(self):
        raise NotImplementedError
   
class D1:
    def __init__(self, lr_d=0.00003, k_d=20, batch=32):
        self.DNet = DNet(3).to(device)
        self.k = k_d
        self.lr = lr_d
        self.batch = batch
        self.joint_buffer = []
        self.marginal_buffer = []
        self.optimizer = optim.Adam(self.DNet.dnet.parameters(), lr=self.lr)
        # Defining the learning rate scheduler
        # self.lr_scheduler = scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)
        self.BCE = nn.BCELoss()

    def clear(self):
        self.joint_buffer = []
        self.marginal_buffer = []
    
    def update(self):
        zeros = torch.zeros(32, 1).to(device)
        ones = torch.ones(32, 1).to(device)
        for _ in range(self.k):
            index = np.random.randint(0, len(self.joint_buffer) - self.batch)
            S = torch.FloatTensor(np.array(self.joint_buffer[index:index + self.batch])).detach().to(device)
            S_target = torch.FloatTensor(np.array(self.marginal_buffer[index:index + self.batch])).detach().to(device)
            loss_1 = self.BCE(torch.sigmoid(self.DNet.dnet(S)), ones).mean()
            loss_0 = self.BCE(torch.sigmoid(self.DNet.dnet(S_target)), zeros).mean()
            loss_D = loss_0 + loss_1
            self.optimizer.zero_grad()
            loss_D.backward()
            self.optimizer.step()
        self.clear()
        # Updated learning rate
        # self.lr_scheduler.step()
    
    def caculate(self, Yt, St, St_1): 
        with torch.no_grad():
            Yt = torch.FloatTensor([Yt]).to(device) 
            St = torch.FloatTensor([St]).to(device) 
            St_1 = torch.FloatTensor([St_1]).to(device) 
            in_put = torch.tensor([Yt, St, St_1]).to(device) 
            predict = self.DNet.dnet(in_put)
        return predict.item()