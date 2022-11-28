from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np

class SIRD:
    def __init__(self, initial_conditions, static_parameters,time_delta, n_var = 1):
        self.initial_conditions = initial_conditions
        self.static_parameters = static_parameters
        self.start_time = time_delta[0]
        self.end_time = time_delta[1]

        assert(len(self.initial_conditions) == (3 + n_var), f"number of initial conditions does not fit with expected variations, got: {n_var}")

        self.initial_conditions_keys = []
        for key,val in self.initial_conditions.items():
            setattr(self, key, val)
            self.initial_conditions_keys.append(key)

        for key,val in self.static_parameters.items():
            setattr(self, key, val)

        # self.N = self.S

        self.get_end_condition_dict()


    def vectorfield(self, w, t, p):
        """
        Defines the differential equations for the coupled spring-mass system.

        Arguments:
            w :  vector of the state variables:
                    w = [x1,y1,x2,y2]
            t :  time
            p :  vector of the parameters:
                    p = [m1,m2,k1,k2,L1,L2,b1,b2]
        """
        S, I_a, I_b, R, D = w
        alpha_a, alpha_b, beta_a, beta_b, gamma_a, gamma_b = p
        t_b = 25
        # Create f = (x1',y1',x2',y2'):
        f = [ - (alpha_a/(self.N) * I_a + alpha_b/(self.N) * I_b ) * S,
            (alpha_a/(self.N)) * S * I_a - beta_a *I_a - gamma_a*I_a,
            (alpha_b/(self.N)) * S * I_b - beta_b *I_b - gamma_b*I_b,
            beta_a * I_a + beta_b * I_b,
            gamma_a * I_a + gamma_b * I_b,
            ]
        return f

    def calc_end_conditions(self, numpoints=None):
        starttime = self.start_time
        stoptime = self.end_time

        if numpoints is None:
            numpoints = stoptime

        # ODE solver parametersq
        abserr = 1.0e-8
        relerr = 1.0e-6
        stoptime = stoptime # in days
        numpoints = numpoints # 1 point per day

        # Create the time samples for the output of the ODE solver.
        # I use a large number of points, only because I want to make
        # a plot of the solution that looks nice.
        # t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]
        self.t = np.linspace(starttime, stoptime, numpoints, endpoint=True)
        # print(t)
        # Pack up the parameters and initial conditions:

        
        p = [val for key, val in self.static_parameters.items()]
        w0 = [val for key, val in self.initial_conditions.items() if not(key is "N") ]

        # p = [alpha_a, alpha_b, beta_a, beta_a, gamma_a, gamma_b]
        # w0 = [S, I_a, I_b, R, D]

        # Call the ODE solver.
        self.all_conditions = odeint(self.vectorfield, w0, self.t, args=(p,),
                    atol=abserr, rtol=relerr)

        self.end_conditions_vals = self.all_conditions[-1, :]
        return self.all_conditions
    
    def get_end_condition_dict(self):
        self.calc_end_conditions()
        self.end_conditions = dict(zip(self.initial_conditions_keys, np.r_[self.N,self.end_conditions_vals]))
        return self.end_conditions



# Parameter values
# Masses:
A = 1.5
beta_a = 0.05
alpha_a = A * beta_a

static_params_1 = {
    "alpha_a": alpha_a,
    "alpha_b": 0.0,
    "beta_a": beta_a,
    "beta_b": 0.000,
    "gamma_a": 0.000,
    "gamma_b": 0.000
}

static_params_2 = static_params_1.copy()
B = 2
beta_b = 0.05
alpha_b = B * beta_b
static_params_2["alpha_b"] = alpha_b
static_params_2["beta_b"] = beta_b
static_params_2["gamma_b"] = 0.000
  
# print(alpha/beta)

# Initial conditions
# x1 and x2 are the initial displacements; y1 and y2 are the initial velocities
I_a = 10
I_b = 0
D = 0
R = 0
S = 500000 - (I_a + I_b + D + R)
T_init = 0

init_condtions_1 = {
    "N": S,
    "S": S,
    "I_a": I_a,
    "I_b": I_b,
    "R": R,
    "D": D,
}

sird_model_1 = SIRD(static_parameters=static_params_1, initial_conditions=init_condtions_1,time_delta = [0,150], n_var=2)
init_condtions_2 = sird_model_1.end_conditions
init_condtions_2["I_b"] += 3
init_condtions_2["S"] -= 3


sird_model_2 = SIRD(static_parameters=static_params_2, initial_conditions=init_condtions_2, time_delta = [150,365*2], n_var=2)


combined_model = np.vstack([sird_model_1.all_conditions, sird_model_2.all_conditions])
combined_time = np.hstack([sird_model_1.t,sird_model_2.t])
plot_legend = [key for key, vel in sird_model_1.initial_conditions.items() if not (key is "N")]



# løbende integral -> fold med kumuleret gaussisk kernel eller kernel der representere fordeling af sygdoms-forløb i tid

print("total of each ",(combined_model[-1,:]).astype(np.int32) )
print("total ",np.sum(combined_model[-1,:]) )


plt.figure()
plt.plot(combined_time, combined_model[:,1:3])

plt.legend(plot_legend[1:3])
plt.show()