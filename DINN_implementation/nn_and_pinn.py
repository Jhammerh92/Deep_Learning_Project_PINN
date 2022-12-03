# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 15:48:55 2022

@author: willi
"""

import numpy as np
import deepxde as dde

from SIRD_normal_nn import SIRD_net
import ODE_SIR
from SIRD_deepxde_class import SIRD_deepxde_net
from SIRD_deepxde_class import Plot

seed = 0
np.random.seed(seed)
dde.config.set_random_seed(seed)

alpha_real = 0.2
beta_real = 0.05
gamma_real = 0.01


solver = ODE_SIR.ODESolver()
t_synth, wsol_synth, N = solver.solve_SIRD(alpha_real, beta_real, gamma_real)
t_bool = t_synth  < 85
t, wsol = t_synth[t_bool], wsol_synth[t_bool]
wsol = solver.add_noise(wsol, scale_pct=0.05)



net = SIRD_net(t, wsol, init_num_people=solver.init_num_people)
net.train()
net.plot(t_synth, wsol_synth)


model = SIRD_deepxde_net(t, wsol)
model.init_model(print_every=1000)
model.train_model(iterations=10000)

alpha_nn, beta_nn, gamma_nn = model.get_best_params()
t_nn_param, wsol_nn_param, N_nn_param = solver.solve_SIRD(alpha_nn, beta_nn, gamma_nn)

