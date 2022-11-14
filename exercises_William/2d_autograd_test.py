# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 10:21:45 2022

@author: willi
"""


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

num_points = 100
xs = np.linspace(0,1,num_points)
ys = np.linspace(0,1,num_points)


rows = np.array([[-1], [-1]])
for i in range(len(xs)):
    row = np.vstack((np.full((num_points),xs[i]), ys))
    rows = np.concatenate((rows, row), axis=1)

rows = np.array(rows)
rows = rows[:,1:]
domain_points = torch.tensor(rows, requires_grad=True)
domain_points = domain_points.float()

n = 2
k0 = 2*np.pi*n

def u(x, y):
    return torch.sin(k0*x)*torch.sin(k0*y)

def f(x, y):
    return torch.sin(x)*torch.sin(y) 


solution = u(domain_points[0,:], domain_points[1,:])

sol2d = solution.reshape(num_points,num_points)

sol2d = sol2d.detach().numpy()


X, Y = np.meshgrid(xs, ys)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, sol2d, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

plt.imshow(sol2d, cmap='hot', interpolation='nearest')
plt.show()


ud = grad(solution, domain_points, 
            grad_outputs = torch.ones_like(solution),
            create_graph=True)[0]

udd = grad(ud, domain_points, 
            grad_outputs = torch.ones_like(ud),
            create_graph=True)[0]

uxx = udd[0,:]
uyy = udd[1,:]

plt.plot(domain_points[0,:].detach(), uxx.detach())


###########
x_test = torch.linspace(-3, 3, 100, requires_grad=True)
x_test = x_test[torch.randperm(len(x_test))]
y_test = torch.linspace(-3, 3, 100, requires_grad=True)
y_test = y_test[torch.randperm(len(y_test))]

def f_test(x,y):
    return torch.sin(x) + 2*y#torch.cos(y)
    
z = f_test(x_test, y_test)
# z.backward(torch.ones(z.shape))

fx = grad(z, x_test, 
            grad_outputs = torch.ones_like(z),
            create_graph=True)[0]
fy = grad(z, y_test, 
            grad_outputs = torch.ones_like(z),
            create_graph=True)[0]


plt.scatter(x_test.detach(), torch.cos(x_test).detach(), label='sin(x)')
# plt.plot(x_test.detach(), x_test.grad)
# plt.plot(y_test.detach(), y_test.grad)
plt.scatter(x_test.detach(), fx.detach(), label='fx')
plt.scatter(y_test.detach(), fy.detach(), label='fy')
plt.legend()