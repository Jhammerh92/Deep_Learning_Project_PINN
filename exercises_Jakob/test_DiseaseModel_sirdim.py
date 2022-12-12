from DiseaseModel import SIRDIm, GeneralModelSolver
import matplotlib.pyplot as plt



static_parameters = {"alpha": 0.11, #0.13
                "beta": 0.08, #0.09
                "gamma": 0.001, #0.001
                "kappa": 1, # 0.05
                }

initial_conditions = {"S": 100000,
            "I": 10,
            "R": 0,
            "D": 0,
            "Im": 0,
            }

#time_delta

# SIR = DiseaseModel(init_cond, static_params, pde)
sirdim = SIRDIm() # init_conditions and static parameters can also be put in here
solver = GeneralModelSolver(sirdim) # uses model to solve by new params


print(sirdim)
sirdim.initialize(initial_conditions, static_parameters, [0,2000])
print(sirdim)

t_synth, sol_synth = sirdim.simulate()

# try:
#     t, sol = solver(alpha=0.066, beta=0.021, gamma=0.001) # doesn't accept parameters that does not match the model
# except:
#     print("didn't work!")

# t, sol = solver(0.066, 0.021) # solver class used here
# t, sol = solver(alpha=0.066, beta=0.021) # solver class used here

fig = plt.figure()
ax = fig.add_subplot(211)
ax.set_title("SIRD")
ax.plot(t_synth, sol_synth[:,:-1])
ax.legend(sirdim.initial_conditions_keys)

ax = fig.add_subplot(212)
ax.set_title("Immunity")
ax.plot(t_synth, sol_synth[:,-1])
ax.legend(sirdim.initial_conditions_keys[-1])
ax.set_ylim([0,1])
# plt.gca().set_prop_cycle(None)
# plt.plot(t, sol, '--')
plt.show()