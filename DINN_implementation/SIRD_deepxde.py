# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 11:39:05 2022

@author: willi
"""
import deepxde as dde
import numpy as np
import torch

alpha_real = 0.2
beta_real = 0.05
gamma_real = 0.01

import ODE_SIR
solver = ODE_SIR.ODESolver()
t, wsol, N = solver.solve_SIRD(alpha_real, beta_real, gamma_real)
solver.plot_SIRD(t, wsol)

timestep = 150
t, wsol = t[:timestep*4], wsol[:timestep*4]

S_sol, I_sol, R_sol, D_sol = wsol[:,0], wsol[:,1], wsol[:,2], wsol[:,3]
init_num_people = np.sum(wsol[0,:])
S_sol, I_sol, R_sol, D_sol = S_sol/init_num_people, I_sol/init_num_people, R_sol/init_num_people, D_sol/init_num_people

timedomain = dde.geometry.TimeDomain(0, max(t))

alpha = dde.Variable(0.1)
beta = dde.Variable(0.1)
gamma = dde.Variable(0.1)

def pde(t, y):
    S, I = y[:, 0:1], y[:, 1:2]
    
    dS_t = dde.grad.jacobian(y, t, i=0)
    dI_t = dde.grad.jacobian(y, t, i=1)
    dR_t = dde.grad.jacobian(y, t, i=2)
    dD_t = dde.grad.jacobian(y, t, i=3)
    
    return [dS_t + alpha*S*I, 
           dI_t - alpha*S*I + beta*I + gamma*I,
           dR_t - beta*I,
           dD_t - gamma*I]

def boundary(_, on_initial):
    return on_initial

# Initial conditions
ic_S = dde.icbc.IC(timedomain, lambda X: torch.tensor(S_sol[0]).reshape(1,1), boundary, component=0)
ic_I = dde.icbc.IC(timedomain, lambda X: torch.tensor(I_sol[0]).reshape(1,1), boundary, component=1)
ic_R = dde.icbc.IC(timedomain, lambda X: torch.tensor(R_sol[0]).reshape(1,1), boundary, component=2)
ic_D = dde.icbc.IC(timedomain, lambda X: torch.tensor(D_sol[0]).reshape(1,1), boundary, component=3)

# Test points
observe_S = dde.icbc.PointSetBC(t.reshape(len(t), 1), S_sol.reshape(len(S_sol), 1), component=0)
observe_I = dde.icbc.PointSetBC(t.reshape(len(t), 1), I_sol.reshape(len(I_sol), 1), component=1)
observe_R = dde.icbc.PointSetBC(t.reshape(len(t), 1), R_sol.reshape(len(R_sol), 1), component=2)
observe_D = dde.icbc.PointSetBC(t.reshape(len(t), 1), D_sol.reshape(len(D_sol), 1), component=3)

data = dde.data.PDE(
    timedomain,
    pde,
    [ic_S, ic_I, ic_R, ic_D, 
     observe_S, observe_I,
     observe_R,observe_D
     ],
    num_domain=50,
    num_boundary=10,
    anchors=t.reshape(len(t), 1),
)


layer_size = [1] + [32] * 3 + [4]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)
variables = [alpha, beta, gamma]
model.compile("adam", lr=0.001, #metrics=["l2 relative error"], 
              external_trainable_variables=variables)

variable = dde.callbacks.VariableValue(variables, period=100)
losshistory, train_state = model.train(iterations=7500, callbacks=[variable])

dde.saveplot(losshistory, train_state, issave=False, isplot=True)