# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 14:00:19 2022

@author: willi
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

def sin(x):
    return torch.sin(x)


x = torch.linspace(-torch.pi, torch.pi, 5000, requires_grad=True)
y = torch.sin(x)
y.backward(torch.ones_like(x))

plt.plot(x.detach().numpy(), y.detach().numpy(), label='sin(x)')
plt.plot(x.detach().numpy(), x.grad.detach().numpy(), label='cos(x)') # print derivative of sin(x)
plt.legend()
plt.show()
##########################

x = torch.linspace(-3, 3, 5000, requires_grad=True)
y = 2*x**2+2*x+2
y.backward(torch.ones_like(x))

plt.plot(x.detach().numpy(), y.detach().numpy(), label='f')
plt.plot(x.detach().numpy(), x.grad.detach().numpy(), label="f'") # print derivative of sin(x)
plt.legend()
plt.grid()
plt.title('2x^2+2x+2')
plt.show()

##########################
xs = torch.linspace(-5, 5, steps=100, requires_grad=True)
ys = torch.linspace(-5, 5, steps=100, requires_grad=True)

x, y = torch.meshgrid(xs, ys, indexing='xy')
x.retain_grad()
y.retain_grad()
z = x**2 + y**2

z.backward(torch.ones_like(z))
ax = plt.axes(projection='3d')
ax.plot_surface(x.detach().numpy(), y.detach().numpy(), z.detach().numpy())
plt.show()

x.grad
y.grad


