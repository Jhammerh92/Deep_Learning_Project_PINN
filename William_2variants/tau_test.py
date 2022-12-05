# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 19:22:07 2022

@author: willi
"""

import deepxde as dde
import numpy as np
from deepxde.backend import pytorch
import torch
import matplotlib.pyplot as plt
from SIRD_deepxde_class_tau import SIRD_deepxde_net
from SIRD_deepxde_class_tau import Plot

seed = 0
np.random.seed(seed)
dde.config.set_random_seed(seed)

tau = 1.4
# tau = sqrt(A)
alpha = 0.7 # infection rate
beta = alpha / tau # recovery rate
alpha /= tau
beta /= tau
gamma = 0.0005

print(alpha, beta)


import ODE_SIR_tau
solver = ODE_SIR_tau.ODESolver()
t_synth, wsol_synth, N = solver.solve_SIRD(alpha, beta, gamma)
solver.plot_SIRD(t_synth, wsol_synth)


max_timestep = 100
t_bool = t_synth < max_timestep
t = t_synth[t_bool]
wsol = wsol_synth[t_bool]



model = SIRD_deepxde_net(t, wsol)
model.init_model(print_every=1000)
model.train_model(iterations=10000)

alpha_nn, tau_nn, gamma_nn = model.get_best_params()
print(alpha_nn, tau_nn, gamma_nn)

t_nn_param, wsol_nn_param, N_nn_param = solver.solve_SIRD(alpha_nn/tau_nn, alpha_nn/(tau_nn**2), gamma_nn)

solver.plot_SIRD(t_nn_param, wsol_nn_param)

