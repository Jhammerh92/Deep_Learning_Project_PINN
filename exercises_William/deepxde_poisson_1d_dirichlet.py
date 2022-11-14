
#https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/poisson.1d.dirichlet.html

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch
import torch.optim as optim
import torch.nn.init as init
from torch.nn.parameter import Parameter
import torch.nn.functional as F
# from sklearn.metrics import accuracy_score

import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from torch.autograd import grad


x = torch.linspace(-1,1,1000, requires_grad=True)
# x = x[torch.randperm(len(x))]

def u(x):
    return torch.sin(torch.pi*x)

def f(x):
    return (torch.pi)**2*torch.sin(torch.pi*x)

###########
boundary_points = torch.tensor([-1,1]).float()
domain_points = x[1:-1]
x_valid = torch.linspace(-1,1,100)

###########
# define network
class Net(nn.Module):

    def __init__(self, num_hidden, num_features, num_output):
        super(Net, self).__init__()  
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
        self.activation = torch.nn.Tanh()
        # self.activation = torch.nn.ReLU()
        # self.activation = torch.nn.Sigmoid()
        self.dropout = nn.Dropout(0.25)
        self.batchnorm = nn.BatchNorm1d(num_hidden, affine=False)
        
    def forward(self, x):
        x = F.linear(x, self.W_1, self.b_1)
        x = self.activation(x)
        # x = self.dropout(x) 
        # x = self.batchnorm(x)
        
        x = F.linear(x, self.W_2, self.b_2)
        x = self.activation(x)
        # x = self.batchnorm(x)
        # x = self.dropout(x)
        
        x = F.linear(x, self.W_2, self.b_2)
        x = self.activation(x)
        # x = self.dropout(x)
        # x = self.batchnorm(x)
        
        x = F.linear(x, self.W_3, self.b_3)
        # x = self.activation(x)
        return x
    
num_hidden, num_features, num_output = 50, 1, 1

net = Net(num_hidden, num_features, num_output)

optimizer = optim.Adam(net.parameters(), lr=0.001)
criterion = nn.MSELoss()

num_epochs = 1000
x_train = domain_points.reshape(len(domain_points),1)
# x_valid = x_valid.reshape(len(x_valid), 1)
get_slice = lambda i, size: range(i * size, (i + 1) * size)
num_samples_train = len(x_train)
batch_size = 100
num_batches_train = num_samples_train // batch_size
num_samples_valid = len(x_valid)
num_batches_valid = num_samples_valid // batch_size

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
        x_batch = x_train[slce]
        output = net(x_batch)
        
        ux = grad(output, x_batch, 
                    grad_outputs = torch.ones_like(x_batch),
                    create_graph=True)[0]
        uxx = grad(ux, x_batch, 
                    grad_outputs = torch.ones_like(x_batch),
                    create_graph=True)[0]
        
        nuxx = -uxx
        
        boundary_output = net(boundary_points.reshape(len(boundary_points),1))
        
        output = torch.cat((nuxx, boundary_output))
        
        # compute gradients given loss
        target_batch = torch.cat(
                        (f(x_batch),
                        torch.tensor([0, 0]).reshape(2,1)))
        
        batch_loss = criterion(output, target_batch)
        batch_loss.backward(retain_graph=True)
        
        optimizer.step()
        
        cur_loss += batch_loss   
    losses.append(cur_loss / batch_size)
    
    # net.eval()
    # ## Evaluate training
    # train_preds, train_targs = [], []
    # for i in range(num_batches_train):
    #     slce = get_slice(i, batch_size)
    #     x_batch = x_train[slce]
    #     output = net(x_batch)
        
    #     train_targs += list(u(x_batch).detach().numpy())
    #     train_preds += list(output.detach().numpy())
    
    # ### Evaluate validation
    # val_preds, val_targs = [], []
    # for i in range(num_batches_valid):
    #     slce = get_slice(i, batch_size)
    #     x_batch = x_valid[slce]
        
    #     output = net(x_batch.reshape(len(x_valid[slce]),1))
    #     val_targs += list(u(x_batch).numpy())
    #     val_preds += list(output.data.numpy())
        
    print("Epoch %2i : Train Loss %f" % (
            epoch+1, losses[-1]))
    
    
###############
solution = u(x)
plt.scatter(x.detach().numpy(), solution.detach().numpy())
plt.show()

###############
# val = net(x.reshape(len(x),1))
# size = 1
# plt.scatter(x.detach().numpy().flatten(), 
#             u(x).detach().numpy().flatten(),
#             s=size, label='True')
# plt.scatter(x.detach().numpy().flatten(), 
#             val.detach().numpy().flatten(),
#             s=size, label='Network')
# plt.title('Train data')
# plt.legend()
# # plt.show()
###############


###############
val = net(x.reshape(len(x),1))
plt.plot(x.detach().numpy().flatten(), 
            u(x).detach().numpy().flatten(),
            label='True')
plt.plot(x.detach().numpy().flatten(), 
            val.detach().numpy().flatten(), 
            label='Network')
plt.title('Train data')
plt.legend()
plt.show()
###############