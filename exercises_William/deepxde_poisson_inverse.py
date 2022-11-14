
# https://deepxde.readthedocs.io/en/latest/demos/pinn_inverse/elliptic.inverse.field.html

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch.optim as optim
import torch.nn.init as init
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.autograd import grad


def u(x):
    return torch.sin(torch.pi*x)

def u_np(x):
    return np.sin(np.pi*x)

def u_bc(x):
    return torch.zeros(x.shape)

# def f(x, t):
#     return torch.zeros(x.shape)

num_points = 100
xs_start, xs_end = -1, 1
xs = np.linspace(xs_start,xs_end,num_points)[1:-1]
xs = torch.tensor(xs[torch.randperm(len(xs))],requires_grad=True).float()

# num_bc_points = 1000
# bc_points_x = torch.cat(
#                 (torch.ones(num_bc_points)*(xs_start),
#                  torch.ones(num_bc_points)*(xs_end)))
bc_points_x = torch.tensor([-1,1]).float()

num_test_points = 100
test_points = np.linspace(xs_start,xs_end,num_test_points)[1:-1]
test_points = torch.tensor(test_points[torch.randperm(len(test_points))],requires_grad=True).float()
test_points_sol = u(test_points)

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
        self.dropout = nn.Dropout(0.1)
        self.batchnorm = nn.BatchNorm1d(num_hidden, affine=False)
        
    def forward(self, x):
        # x = torch.stack((x, torch.empty(0)),axis=1)
        x = x.reshape(x.shape[0],1)
        x = F.linear(x, self.W_1, self.b_1)
        x = self.activation(x)
        # x = self.dropout(x) 
        # x = self.batchnorm(x)
        
        x = F.linear(x, self.W_2, self.b_2)
        x = self.activation(x)
        # x = self.dropout(x)
        # x = self.batchnorm(x)
        
        
        x = F.linear(x, self.W_2, self.b_2)
        x = self.activation(x)
        # x = self.dropout(x)
        # x = self.batchnorm(x)
        
        x = F.linear(x, self.W_2, self.b_2)
        x = self.activation(x)
        
        x = F.linear(x, self.W_2, self.b_2)
        x = self.activation(x)
        
        x = F.linear(x, self.W_2, self.b_2)
        x = self.activation(x)
        # x = self.dropout(x) 
        
        # x = F.linear(x, self.W_2, self.b_2)
        # x = self.activation(x)
        
        x = F.linear(x, self.W_3, self.b_3)
        # x = self.activation(x)
        return x
    
# num_hidden, num_features, num_output = 50, 2, 1
num_hidden, num_features, num_output = 50, 1, 1

net = Net(num_hidden, num_features, num_output)
net_q = Net(num_hidden, num_features, num_output)

optimizer = optim.Adam(net.parameters(), lr=0.01)
optimizer_q = optim.Adam(net_q.parameters(), lr=0.01)
criterion = nn.MSELoss()

num_epochs = 200
x_train = xs #reshape(len(domain_points),2)

get_slice = lambda i, size: range(i * size, (i + 1) * size)
num_samples_train = x_train.shape[0]
batch_size = 10
num_batches_train = num_samples_train // batch_size

train_acc, train_loss = [], []
valid_acc, valid_loss = [], []
test_acc, test_loss = [], []

losses = []
def calc_grad(inp, out):
    return grad(out, inp, 
                grad_outputs = torch.ones_like(out),
                create_graph=True)[0]

for epoch in range(num_epochs):
    # Forward -> Backprob -> Update params
    ## Train
    cur_loss = 0
    net.train()
    
    for i in range(num_batches_train):
        optimizer.zero_grad()
        optimizer_q.zero_grad()
        slce = get_slice(i, batch_size)
        x_batch = xs[slce]
        u_net = net(x_batch)
        u_net = u_net.reshape(u_net.shape[0])
        
        ux = calc_grad(x_batch, u_net)
        uxx = calc_grad(x_batch, ux)

        q_net = net_q(x_batch)
        q_net = q_net.reshape(q_net.shape[0])

        lhs_pde = uxx - q_net
        # compute gradients given loss
        # domain_target = f(x_batch, t_batch)
        
        bc_output = net(bc_points_x)
        bc_output = bc_output.reshape(bc_output.shape[0])
        
        net_test_sol = net(test_points)
        net_test_sol = net_test_sol.reshape(net_test_sol.shape[0])
        
        output = torch.cat((lhs_pde, 
                            bc_output,
                            net_test_sol))
        
        # compute gradients given loss
        # domain_target = f(x_batch, t_batch)
        domain_target = torch.zeros(lhs_pde.shape)
        bc_target = u_bc(bc_points_x) #torch.empty(0)
        
        target_batch = torch.cat((domain_target,
                                  bc_target,
                                  test_points_sol))
        
        batch_loss = criterion(output.flatten(), 
                               target_batch.flatten(),
                               )
        batch_loss.backward(retain_graph=True)
        
        optimizer.step()
        optimizer_q.step()
        
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
        
    if epoch%10==0:
        print("Epoch %2i : Train Loss %f" % (
            epoch+1, losses[-1]))


###############################################################################
xs_np = xs.detach().numpy()
vals = np.sin(np.pi*xs_np)

plt.scatter(xs_np, vals, label='exact')
# plt.title('Exact')
# plt.show()

######
nn_output = net(xs)

plt.scatter(xs.detach().numpy(), nn_output.detach().numpy(),label='network')
plt.title('Network result')
plt.legend()
plt.show()
