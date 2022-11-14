# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 14:44:51 2022

@author: willi
"""

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

def sin(x):
    return torch.sin(x)

# def sin(x):
#     return x**2+1


# x_vals = np.linspace(-3,3,10000)
low, high = -3, 3
x_vals = np.random.default_rng().uniform(low,high,10000)
x = torch.tensor(x_vals, requires_grad=True)
x_valid = np.random.default_rng().uniform(low,high,1000)
x_valid = torch.tensor(x_valid, requires_grad=True)

targets_train = sin(x)
targets_valid = sin(x_valid)

targets_train.backward(torch.ones_like(targets_train),
                       retain_graph=True)
xgrad = x.grad
targets_train = torch.cat((targets_train,xgrad))


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
        self.W_22 = Parameter(init.xavier_normal_(torch.Tensor(num_hidden, num_hidden)))
        self.b_22 = Parameter(init.constant_(torch.Tensor(num_hidden), 0))
        
        # hidden layer
        self.W_3 = Parameter(init.xavier_normal_(torch.Tensor(num_output, num_hidden)))
        self.b_3 = Parameter(init.constant_(torch.Tensor(num_output), 0))
        # define activation function in constructor
        # self.activation = torch.nn.ELU()
        self.activation = torch.nn.Tanh()
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = F.linear(x, self.W_1, self.b_1)
        x = self.activation(x)
        # x = self.dropout(x) 
        x = F.linear(x, self.W_2, self.b_2)
        x = self.activation(x)
        # x = self.dropout(x)
        # x = F.linear(x, self.W_22, self.b_22)
        # x = self.activation(x)
        # x = self.dropout(x)
        x = F.linear(x, self.W_3, self.b_3)
        # x = self.activation(x)
        return x

num_hidden, num_features, num_output = 100, 1, 1

net = Net(num_hidden, num_features, num_output)

optimizer = optim.SGD(net.parameters(), lr=0.01)
criterion = nn.MSELoss()

num_epochs = 500
x_train = x.reshape(len(x),1).float()
x_valid = x_valid.reshape(len(x_valid), 1).float()
get_slice = lambda i, size: range(i * size, (i + 1) * size)
num_samples_train = len(x_train)
batch_size = 10
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
        output = net(x_train[slce])
        
        # compute gradients given loss
        target_batch = targets_train[slce]
        target_batch = target_batch.reshape(len(target_batch), 1).float()
        # target_batch = sin(x_train[slce])
        batch_loss = criterion(output, target_batch)
        batch_loss.backward(retain_graph=True)
        optimizer.step()
        
        cur_loss += batch_loss   
    losses.append(cur_loss / batch_size)
    
    net.eval()
    ### Evaluate training
    train_preds, train_targs = [], []
    for i in range(num_batches_train):
        slce = get_slice(i, batch_size)
        output = net(x_train[slce])
        
        train_targs += list(targets_train[slce].detach().numpy())
        train_preds += list(output.detach().numpy())
    
    ### Evaluate validation
    val_preds, val_targs = [], []
    for i in range(num_batches_valid):
        slce = get_slice(i, batch_size)
        
        output = net(x_valid[slce])
        val_targs += list(targets_valid[slce].detach().numpy())
        val_preds += list(output.data.numpy())
        
    train_err_cur = np.sqrt(metrics.mean_squared_error(np.array(train_targs), 
                                   np.array(train_preds).flatten()))
    valid_err_cur = np.sqrt(metrics.mean_squared_error(np.array(val_targs), 
                                   np.array(val_preds)))
    
    train_acc.append(train_err_cur)
    valid_acc.append(valid_err_cur)
    
    if epoch % 10 == 0:
        print("Epoch %2i : Train Loss %f , Train acc %f, Valid acc %f" % (
                epoch+1, losses[-1], train_err_cur, valid_err_cur))

epoch = np.arange(len(train_acc))
plt.figure()
plt.plot(epoch, train_acc, 'r', epoch, valid_acc, 'b')
plt.legend(['Train Accucary','Validation Accuracy'])
plt.xlabel('Updates'), plt.ylabel('Acc')

plt.show()


###############
val = net(x_valid)
size = 0.1
plt.scatter(x_train.detach().numpy().flatten(), targets_train.detach().numpy().flatten(),
            s=size)
plt.scatter(x_train.detach().numpy().flatten(), val.detach().numpy().flatten(),
            s=size)
plt.title('Train data')
plt.show()
###############


###############
x_out = np.random.default_rng().uniform(0,5,10000)
target = sin(x_out)
x_out = torch.from_numpy(x_out)
x_out = x_out.float().reshape(len(x_out),1)

val = net(x_out)
size = 0.1
plt.scatter(x_out.detach().numpy().flatten(), target.flatten(),
            s=size)
plt.scatter(x_out.detach().numpy().flatten(), val.detach().numpy().flatten(),
            s=size)
plt.title('Outside of domain')
###############