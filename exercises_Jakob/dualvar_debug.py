import deepxde as dde
import numpy as np
from deepxde.backend import pytorch
import torch
import matplotlib.pyplot as plt
from SIRD_dualvar_simplified_deepxde_class import SIRD_deepxde_net, Plot
# from SIRD_d import SIRD_deepxde_net, Plot
# from SIRD_deepxde_class import Plot
# from ODE_SIR_copy import ODESolver

from ODE_SIRD_reinfection_class import SIRD2VAR, static_params_1, init_condtions_1, t_intro_var2, n_intro_var2

seed = 0
np.random.seed(seed)
dde.config.set_random_seed(seed)

solver = SIRD2VAR.solve_by_params_alpha_tau

# import ODE_SIR
# solver = ODE_SIR.ODESolver()
# t_synth, wsol_synth, N = solver.solve_SIRD(alpha_real, beta_real, gamma_real)
# solver.plot_SIRD(t_synth, wsol_synth)

time_delta = [0,2*365]
# import ODE_SIRD_reinfection_class as sird2

sird_model = SIRD2VAR(init_condtions_1, static_params_1, time_delta, t_intro_var2, n_intro_var2)
t_synth, wsol_synth = sird_model.get_wsol_as_SIRD()
t_synth, wsol_synth_full = sird_model.get_wsol()
sird_model.plot_wsol()
sird_model.plot_sird()

# keep this even if not subsetting
t = t_synth
wsol = wsol_synth

# subset
# max_timestep = 300
# t_bool = t_synth < max_timestep
# t = t_synth[t_bool]
# wsol = wsol_synth[t_bool]

# keep this even if not subsetting
t = t_synth
wsol = wsol_synth

# subset
# max_timestep = 300
# t_bool = t_synth < max_timestep
# t = t_synth[t_bool]
# wsol = wsol_synth[t_bool]

model = SIRD_deepxde_net(t, wsol, with_neumann=False, model_name="dualvar_tau", with_softadapt=True)
print(model)

hyper_print_every = 1
model.init_model(print_every=hyper_print_every)

model.train_model(iterations=2000, print_every=hyper_print_every, use_LBFGSB=False)


print(f"Best train step: {model.model.train_state.best_step}")
params_nn = model.get_best_params()
# print('Alpha_a: {}, Alpha_b: {}, Alpha_aa: {}, Alpha_bb: {}, Alpha_ba: {}, Alpha_ab: {}, beta_a: {}, beta_b: {}, gamma_a: {}, gamma_b: {}'.format(*params_nn))
# print('Alpha_a: {}, Alpha_b: {}, beta_a: {}, beta_b: {}, gamma_a: {}, gamma_b: {}'.format(*params_nn))
print('Alpha_a: {}, Alpha_b: {}, tau_a: {}, tau_b: {}, gamma_a: {}, gamma_b: {}'.format(*params_nn))
# alpha_a_nn, alpha_b_nn, beta_a_nn, beta_b_nn, gamma_a_nn, gamma_b_nn = params_nn
t_nn_param, wsol_nn_param, N_nn_param = solver(*params_nn)
# plt.plot(t_nn_param, wsol_nn_param)

# we need to set the synthetic data as it comes from outside the network
# the two functions below sets the synthetic data
model.set_synthetic_data(t_synth, wsol_synth_full) 
model.set_nn_synthetic_data(t_nn_param, wsol_nn_param)

plot = Plot(model) # class that contains plotting functions

plot.show_known_and_prediction()

plt.show()