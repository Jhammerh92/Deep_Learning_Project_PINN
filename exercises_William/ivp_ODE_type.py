
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch.optim as optim
import torch.nn.init as init
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.autograd import grad

num_points = 10000
lambds = np.linspace(-1,1,num_points)
lambds = torch.tensor(lambds[torch.randperm(len(lambds))],requires_grad=True).float()
ts = np.linspace(0,2,num_points)[1:]
ts = torch.tensor(ts[torch.randperm(len(ts))],requires_grad=True).float()

num_iv_points = 100
ivt = torch.zeros(num_iv_points).float()
ivlambd = torch.tensor(np.linspace(-1,1,num_iv_points)).float()

#####
# Rewriting equation gives
# u'-lambda*u=0
#####


u0 = 1

def u(t, lamb):
    return u0*torch.exp(lamb*t)

def f(t, lamb):
    return torch.zeros(t.shape)

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

num_epochs = 1000
x_train = ts #reshape(len(domain_points),2)

get_slice = lambda i, size: range(i * size, (i + 1) * size)
num_samples_train = x_train.shape[0]
batch_size = 1000
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
        t_batch = ts[slce]
        lambd_batch = lambds[slce]
        output = net(t_batch, lambd_batch)
        output = output.reshape(output.shape[0])
        ut = calc_grad(t_batch, output)

        lhs_pde = ut - lambd_batch*output
        
        boundary_output = net(ivt, ivlambd)
        boundary_output = boundary_output.reshape(boundary_output.shape[0])
        
        # lhs_pde = (-uxx.reshape(len(uxx), 1) 
        #            -uyy.reshape(len(uyy), 1) 
        #            -k0**2*output)
        output = torch.cat((lhs_pde, boundary_output))
        
        # compute gradients given loss
        domain_target = f(t_batch, lambd_batch)
        boundary_target = torch.ones(len(boundary_output))*u0
        
        target_batch = torch.cat((domain_target,
                                  boundary_target))
        
        batch_loss = criterion(output.flatten(), target_batch.flatten())
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
        
    # if epoch%100==0:
    print("Epoch %2i : Train Loss %f" % (
            epoch+1, losses[-1]))
    
# num_points = 100
# xs = np.linspace(0,1,num_points)
# ys = np.linspace(0,1,num_points)

# boundary_points = np.array([[-1], [-1]])

# rows = np.array([[-1], [-1]])
# for i in range(len(xs)):
#     if i == 0 or i == len(xs) - 1:
#         row_bp = np.vstack((np.full((num_points-2),xs[i]), ys[1:-1]))
#         boundary_points = np.concatenate((boundary_points, row_bp), axis=1)
#     else:
#         row = np.vstack((np.full((num_points-2),xs[i]), ys[1:-1]))
#         rows = np.concatenate((rows, row), axis=1)

# row = np.vstack((xs[1:-1], np.full((num_points-2),ys[0])))
# boundary_points = np.concatenate((boundary_points, row), axis=1)
# row = np.vstack((xs[1:-1], np.full((num_points-2),ys[-1])))
# boundary_points = np.concatenate((boundary_points, row), axis=1)


# rows = np.array(rows)
# rows = rows[:,1:]
# domain_points = torch.tensor(rows, requires_grad=True)
# domain_points = domain_points.float()

# result = net(domain_points[0,:], domain_points[1,:])

# plt.imshow(result.reshape(100-2,100-2).detach(), cmap='hot', interpolation='nearest')
# plt.show()

def u_plot(t, lamb):
    return u0*np.exp(lamb*t)

vals = []

ts_plot = np.linspace(0,2,1000)
lambds_plot = np.linspace(-1,1, 1000)
for lambd_val in lambds_plot:
    u_val = u_plot(ts_plot, lambd_val)
    vals.append(u_val)

vals = np.array(vals)

# plt.scatter(ts_plot, u_plot)
plt.imshow(vals, cmap='hot', interpolation='nearest')
plt.title('Exact result')
plt.show()

nn_input_t = []
nn_input_lambd = []
for nn_t in ts_plot:
    for nn_lambd in lambds_plot:
        nn_input_t.append(nn_t)
        nn_input_lambd.append(nn_lambd)

nn_input_t = torch.tensor(nn_input_t).float()
nn_input_lambd = torch.tensor(nn_input_lambd).float()

nn_output = net(nn_input_t, nn_input_lambd)

plt.imshow(nn_output.reshape(1000,1000).detach().T, cmap='hot', interpolation='nearest')
plt.title('Network result')
plt.show()