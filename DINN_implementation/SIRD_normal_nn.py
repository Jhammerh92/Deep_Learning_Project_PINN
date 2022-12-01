

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch.optim as optim
import torch.nn.init as init
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self, num_hidden, num_features, num_output, num_hidden_layers):
        super(Net, self).__init__()  
        self.num_hidden_layers = num_hidden_layers
        # input layer
        self.W_1 = Parameter(init.xavier_normal_(torch.Tensor(num_hidden, num_features)))
        self.b_1 = Parameter(init.constant_(torch.Tensor(num_hidden), 0))
        # hidden layer
        self.W_2 = Parameter(init.xavier_normal_(torch.Tensor(num_hidden, num_hidden)))
        self.b_2 = Parameter(init.constant_(torch.Tensor(num_hidden), 0))
        # hidden layer
        self.W_3 = Parameter(init.xavier_normal_(torch.Tensor(num_output, num_hidden)))
        self.b_3 = Parameter(init.constant_(torch.Tensor(num_output), 0))
        # define activation function in constructor
        # self.activation = torch.nn.ELU()
        # self.activation = torch.nn.Tanh()
        # self.activation = torch.nn.ReLU()
        self.activation = torch.nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)
        self.batchnorm = nn.BatchNorm1d(num_hidden, affine=False)
        
    def forward(self, x):
        #x = torch.stack((x,y),axis=1)
        x = F.linear(x, self.W_1, self.b_1)
        x = self.activation(x)
        # x = self.dropout(x) 
        # x = self.batchnorm(x)
        
        for i in range(self.num_hidden_layers):
            x = F.linear(x, self.W_2, self.b_2)
            x = self.activation(x)
        
        x = F.linear(x, self.W_3, self.b_3)
        # x = self.activation(x)
        return x

class SIRD_net:
    def __init__(self, t, wsol):
        t = t
        self.wsol = wsol

        S_sol, I_sol, R_sol, D_sol = wsol[:,0], wsol[:,1], wsol[:,2], wsol[:,3]
        init_num_people = np.sum(wsol[0,:])
        S_sol, I_sol, R_sol, D_sol = S_sol/init_num_people, I_sol/init_num_people, R_sol/init_num_people, D_sol/init_num_people


        self.t = torch.tensor(t, requires_grad=True).reshape(len(t),1).float()
        self.S_sol = torch.tensor(S_sol, requires_grad=True).reshape(len(S_sol),1).float()
        self.I_sol = torch.tensor(I_sol, requires_grad=True).reshape(len(I_sol),1).float()
        self.R_sol = torch.tensor(R_sol, requires_grad=True).reshape(len(R_sol),1).float()
        self.D_sol = torch.tensor(D_sol, requires_grad=True).reshape(len(D_sol),1).float()
        self.init_num_people = torch.tensor(init_num_people).float()
    
    def train(self):
        num_hidden, num_features, num_output = 50, 1, 1
        num_hidden_layers = 5

        self.S_net = Net(num_hidden, num_features, num_output, num_hidden_layers)
        self.I_net = Net(num_hidden, num_features, num_output, num_hidden_layers)
        self.R_net = Net(num_hidden, num_features, num_output, num_hidden_layers)
        self.D_net = Net(num_hidden, num_features, num_output, num_hidden_layers)

        t_train = self.t

        get_slice = lambda i, size: range(i * size, (i + 1) * size)
        num_samples_train = t_train.shape[0]
        batch_size = 100
        num_batches_train = num_samples_train // batch_size

        num_epochs = 2500

        for net, sol in zip([self.S_net, self.I_net, self.R_net, self.D_net], 
                            [self.S_sol, self.I_sol, self.R_sol, self.D_sol]):
            optimizer = optim.Adam(net.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            train_acc, train_loss = [], []
            valid_acc, valid_loss = [], []
            test_acc, test_loss = [], []
            
            losses = []
            for epoch in range(num_epochs):
                # Forward -> Backprob -> Update params
                ## Train
                cur_loss = 0
                net.train()
                for i in range(num_batches_train):
                    optimizer.zero_grad()
            
                    slce = get_slice(i, batch_size)
                    t_batch = t_train[slce]
                    target = sol[slce]
                    
                    output = net(t_batch)
                    output = output.reshape(output.shape[0])
                    
                    batch_loss = criterion(output.flatten(), target.flatten())
                    
                    batch_loss.backward()#retain_graph=True)
                    
                    optimizer.step()
                    cur_loss += batch_loss   
                losses.append(cur_loss / batch_size)
                    
                if epoch%100==0:
                    print("Epoch %2i : Train Loss %f" % (
                        epoch+1, losses[-1]))
    def plot(self, t_synth, wsol_synth):
        plt.figure()
        colors = ['C0','C1','C2','C3']
        for pop_type, net, sol, c in zip(['S','I','R','D'],
                                 [self.S_net, self.I_net, self.R_net, self.D_net], 
                                 [self.S_sol, self.I_sol, self.R_sol, self.D_sol],
                                 colors):
            plt.plot(self.t.flatten().detach(), net(self.t).flatten().detach(),label=f'{pop_type}_pred', color=c)
            plt.plot(self.t.flatten().detach(), sol.flatten().detach(),label=f'{pop_type}_sol', linestyle='--', color=c)

        plt.legend()

        plt.figure()
        for pop_type, net, sol_num, c in zip(['S','I','R','D'],
                                 [self.S_net, self.I_net, self.R_net, self.D_net], 
                                 [0,1,2,3],
                                 colors):

            synth_pred = net(torch.tensor(t_synth).reshape(len(t_synth),1).float())*self.init_num_people
            plt.plot(t_synth, synth_pred.flatten().detach(),label=f'{pop_type}_pred', color=c)
            plt.plot(t_synth, wsol_synth[:,sol_num],label=f'{pop_type}_sol', linestyle='--', color=c)

        plt.legend()
                

# alpha_real = 0.2
# beta_real = 0.05
# gamma_real = 0.01

# import ODE_SIR
# solver = ODE_SIR.ODESolver()
# t_synth, wsol_synth, N = solver.solve_SIRD(alpha_real, beta_real, gamma_real)
# solver.plot_SIRD(t_synth, wsol_synth)

# # keep this even if not subsetting
# t = t_synth
# wsol = wsol_synth

# # subset
# max_timestep = 120 # Try 85, 100, and 120 and see difference
# t_bool = t_synth < max_timestep
# t = t_synth[t_bool]
# wsol = wsol_synth[t_bool]

# net = SIRD_net(t, wsol)
# net.train()
# net.plot(t_synth, wsol_synth)





