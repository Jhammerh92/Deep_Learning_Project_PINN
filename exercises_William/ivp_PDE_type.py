

#########################
# selecting h(x) = sin(x)
#########################

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch.optim as optim
import torch.nn.init as init
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.autograd import grad


c = 2
t0 = 0
def u(x, t):
    return torch.sin(x-c*t)

def u_ic(x):
    return torch.sin(x)

def u_bc(t):
    return torch.sin(-c*t)

def u_np(x, t):
    return np.sin(x-c*t)

def f(x, t):
    return torch.zeros(x.shape)


num_points = 50000
xs = np.linspace(-np.pi,np.pi,num_points)[1:-1]
xs = torch.tensor(xs[torch.randperm(len(xs))],requires_grad=True).float()
ts = np.linspace(0,2,num_points)[1:]
ts = torch.tensor(ts[torch.randperm(len(ts))],requires_grad=True).float()

num_ic_points = 1000
ic_points_x = torch.tensor(np.linspace(-np.pi,np.pi,num_points)).float()
ic_points_t = torch.zeros(ic_points_x.shape).float()

num_bc_points = 1000
bc_points_x = torch.cat(
                (torch.ones(num_bc_points)*(-1)*torch.pi,
                 torch.ones(num_bc_points)*torch.pi))
                 
bc_points_t = torch.cat(
                (torch.tensor(np.linspace(0,2,num_bc_points)).float(),
                torch.tensor(np.linspace(0,2,num_bc_points)).float()))


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
        
    def forward(self, x, y):
        x = torch.stack((x,y),axis=1)
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
num_hidden, num_features, num_output = 50, 2, 1

net = Net(num_hidden, num_features, num_output)

optimizer = optim.Adam(net.parameters(), lr=0.01)
criterion = nn.MSELoss()

num_epochs = 200
x_train = xs #reshape(len(domain_points),2)

get_slice = lambda i, size: range(i * size, (i + 1) * size)
num_samples_train = x_train.shape[0]
batch_size = 5000
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
        slce = get_slice(i, batch_size)
        x_batch = xs[slce]
        t_batch = ts[slce]
        output = net(x_batch, t_batch)
        output = output.reshape(output.shape[0])
        
        ut = calc_grad(t_batch, output)
        ux = calc_grad(x_batch, output)

        # lhs_pde = - ut - c*ux
        lhs_pde = ut
        
        bc_output = net(bc_points_x, bc_points_t)
        bc_output = bc_output.reshape(bc_output.shape[0])
        
        ic_output = net(ic_points_x, ic_points_t)
        ic_output = ic_output.reshape(ic_output.shape[0])
        
        
        output = torch.cat((lhs_pde, bc_output, ic_output))
        
        # compute gradients given loss
        # domain_target = f(x_batch, t_batch)
        domain_target = -c*ux
        # raise Exception('Check input to u_bc and u_ic')
        bc_target = u_bc(bc_points_t)
        ic_target = u_ic(ic_points_x)
        
        target_batch = torch.cat((domain_target,
                                  bc_target,
                                  ic_target))
        
        batch_loss = criterion(output.flatten(), 
                               target_batch.flatten(),
                               )
        batch_loss.backward()#retain_graph=True)
        
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
        
    if epoch%10==0:
        print("Epoch %2i : Train Loss %f" % (
            epoch+1, losses[-1]))
    # if losses[-1] < 0.00005:
    #     print('broke')
    #     break

###############################################################################
vals = []

ts_plot = np.linspace(0,2,1000)
xs_plot = np.linspace(-np.pi,np.pi, 1000)
for xs_val in xs_plot:
    u_val = u_np(ts_plot, xs_val)
    vals.append(u_val)

vals = np.array(vals)

plt.figure()
plt.imshow(vals, cmap='hot', interpolation='nearest')
plt.title('Exact result')
plt.show()
i = 1
plt.scatter(xs_plot, vals[:,i])
plt.title(f'Slice {i}')
# plt.show()

nn_input_t = []
nn_input_x = []
for nn_t in ts_plot:
    for nn_x in xs_plot:
        nn_input_t.append(nn_t)
        nn_input_x.append(nn_x)

nn_input_t = torch.tensor(nn_input_t).float()
nn_input_x = torch.tensor(nn_input_x).float()

nn_output = net(nn_input_x, nn_input_t)
plt.figure()
plt.imshow(nn_output.reshape(1000,1000).detach().T, cmap='hot', interpolation='nearest')
plt.title('Network result')

plt.show()

nn_output.reshape(1000,1000)[:,1]
