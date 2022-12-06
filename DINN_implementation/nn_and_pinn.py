# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 15:48:55 2022

@author: willi
"""

import numpy as np
import deepxde as dde
import matplotlib.pyplot as plt

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


#TODO - make sure that this selects data from best run
values_to_train = ['S']

net = SIRD_net(t, wsol, init_num_people=solver.init_num_people, values_to_train=values_to_train)
net.train()





model = SIRD_deepxde_net(t, wsol, init_num_people=solver.init_num_people, model_name='FuturePrediction')
model.init_model(print_every=100)
model.train_model(iterations=40000, print_every=100)

alpha_nn, beta_nn, gamma_nn = model.get_best_params()
t_nn_param, wsol_nn_param, N_nn_param = solver.solve_SIRD(alpha_nn, beta_nn, gamma_nn)

model.set_synthetic_data(t_synth, wsol_synth) 
model.set_nn_synthetic_data(t_nn_param, wsol_nn_param)





#### PLOT
values_to_plot=values_to_train
plot = Plot(model, values_to_plot=values_to_plot)
# plot.show_known_and_prediction()


linewidth=2
fig, ax = plt.subplots(dpi=300, figsize=(6,6))
ax.set_title('Future prediction')
plot._plot_pred_synthetic(ax, alpha=1, linewidth=linewidth)

alpha=0.3
if 'S' in values_to_plot:
    ax.scatter(t, wsol[:,0],alpha=alpha, label='Train data', color='C0')

if 'I' in values_to_plot:
    ax.scatter(t, wsol[:,1],alpha=alpha, label='Train data', color='C1')

plot._plot_pred_nn(ax, linewidth=linewidth)


net.plot(ax, t_synth, values_to_plot=values_to_plot, linewidth=linewidth)
# ax.vlines(x=85, ymin=model.wsol_synth.min(), ymax=model.wsol_synth.max(), linestyle='--', lw=0.5, color='k', label='t_{cut}')
# TODO - change labels
# TODO - write predicted parameters in this case?
ax.legend()
ax.grid(linestyle=':') #
ax.set_axisbelow(True)
ax.set_xlabel('Time [day]')
ax.set_ylabel('Number of people')
plt.savefig('Future prediction',bbox_inches='tight')




dde.utils.external.plot_loss_history(model.losshistory ,fname='Future prediction loss')
