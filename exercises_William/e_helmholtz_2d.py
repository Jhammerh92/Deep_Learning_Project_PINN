
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
xs = np.linspace(0,1,num_points)[1:-1]
xs = torch.tensor(xs[torch.randperm(len(xs))],requires_grad=True).float()
ys = np.linspace(0,1,num_points)[1:-1]
ys = torch.tensor(ys[torch.randperm(len(ys))],requires_grad=True).float()
bpsx = torch.tensor([0,0,1,1]).float()
bpsy = torch.tensor([0,1,0,1]).float()

n = 2
k0 = 2*np.pi*n

def u(x, y):
    return torch.sin(k0*x)*torch.sin(k0*y)

def f(x, y):
    return k0**2*torch.sin(k0*x)*torch.sin(k0*y) 

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

num_epochs = 3000
x_train = xs #reshape(len(domain_points),2)

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
        x_batch = xs[slce]
        y_batch = ys[slce]
        output = net(x_batch, y_batch)
        
        ux = calc_grad(x_batch, output)
        uxx = calc_grad(x_batch, ux)
        
        uy = calc_grad(y_batch, output)
        uyy = calc_grad(y_batch, uy)
        
        boundary_output = net(bpsx, bpsy)
        
        lhs_pde = (-uxx.reshape(len(uxx), 1) 
                   -uyy.reshape(len(uyy), 1) 
                   -k0**2*output)
        output = torch.cat((lhs_pde, boundary_output))
        
        # compute gradients given loss
        domain_target = f(x_batch, y_batch)
        boundary_target = torch.zeros(len(bpsx))
        
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
    
num_points = 100
xs = np.linspace(0,1,num_points)
ys = np.linspace(0,1,num_points)

boundary_points = np.array([[-1], [-1]])

rows = np.array([[-1], [-1]])
for i in range(len(xs)):
    if i == 0 or i == len(xs) - 1:
        row_bp = np.vstack((np.full((num_points-2),xs[i]), ys[1:-1]))
        boundary_points = np.concatenate((boundary_points, row_bp), axis=1)
    else:
        row = np.vstack((np.full((num_points-2),xs[i]), ys[1:-1]))
        rows = np.concatenate((rows, row), axis=1)

row = np.vstack((xs[1:-1], np.full((num_points-2),ys[0])))
boundary_points = np.concatenate((boundary_points, row), axis=1)
row = np.vstack((xs[1:-1], np.full((num_points-2),ys[-1])))
boundary_points = np.concatenate((boundary_points, row), axis=1)


rows = np.array(rows)
rows = rows[:,1:]
domain_points = torch.tensor(rows, requires_grad=True)
domain_points = domain_points.float()

result = net(domain_points[0,:], domain_points[1,:])

plt.imshow(result.reshape(100-2,100-2).detach(), cmap='hot', interpolation='nearest')
plt.show()

# ###############
solution = u(domain_points[0,:], domain_points[1,:])
plt.imshow(solution.reshape(100-2,100-2).detach(), cmap='hot', interpolation='nearest')
plt.show()

# ###############
# # val = net(x.reshape(len(x),1))
# # size = 1
# # plt.scatter(x.detach().numpy().flatten(), 
# #             u(x).detach().numpy().flatten(),
# #             s=size, label='True')
# # plt.scatter(x.detach().numpy().flatten(), 
# #             val.detach().numpy().flatten(),
# #             s=size, label='Network')
# # plt.title('Train data')
# # plt.legend()
# # # plt.show()
# ###############


# ###############
# val = net(x.reshape(len(x),1))
# plt.plot(x.detach().numpy().flatten(), 
#             u(x).detach().numpy().flatten(),
#             label='True')
# plt.plot(x.detach().numpy().flatten(), 
#             val.detach().numpy().flatten(), 
#             label='Network')
# plt.title('Train data')
# plt.legend()
# plt.show()
###############