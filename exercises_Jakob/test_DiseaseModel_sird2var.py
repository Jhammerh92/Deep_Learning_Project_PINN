from DiseaseModel import SIRD2Var, GeneralModelSolver
import matplotlib.pyplot as plt

initial_conditions = {
    "S": 1000000,
    "Ia": 1,
    "Ib": 0,
    "Ra": 0,
    "Rb": 0,
    "D": 0,
    "Im_a": 0, # should be between 0 and 1
    "Im_b": 0, # should be between 0 and 1
    }
static_parameters = {
    "alpha_a": 0.12,
    "alpha_b": 0.12,
    "beta_a": 0.08,
    "beta_b": 0.08,
    "gamma_a": 0.00,
    "gamma_b": 0.00,
    "kappa_a": 0.2,
    "kappa_b": 0.2,
    }

#time_delta

# SIR = DiseaseModel(init_cond, static_params, pde)
sirdim = SIRD2Var() # init_conditions and static parameters can also be put in here
solver = GeneralModelSolver(sirdim) # uses model to solve by new params


print(sirdim)
sirdim.initialize(initial_conditions, static_parameters, [0,20,2000])
# sirdim.initialize(initial_conditions, static_parameters, [0,100,1000])

print(sirdim)

t_synth, sol_synth = sirdim.simulate(var2_n_introduced_infected=1)

# try:
#     t, sol = solver(alpha=0.066, beta=0.021, gamma=0.001) # doesn't accept parameters that does not match the model
# except:
#     print("didn't work!")

# t, sol = solver(0.066, 0.021) # solver class used here
# t, sol = solver(alpha=0.066, beta=0.021) # solver class used here

fig = plt.figure()
ax = fig.add_subplot(211)
ax.set_title("SIRD")
ax.plot(t_synth, sol_synth[:,:-2])
ax.legend(sirdim.initial_conditions_keys)
ax.plot(t_synth, sol_synth[:,1] + sol_synth[:,2], 'g--')

ax = fig.add_subplot(212)
ax.set_title("Immunity")
ax.plot(t_synth, sol_synth[:,-2:])
ax.legend(sirdim.initial_conditions_keys[-2:])
# plt.gca().set_prop_cycle(None)
# plt.plot(t, sol, '--')
plt.show()