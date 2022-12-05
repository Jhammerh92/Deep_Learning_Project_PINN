from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np

t_intro_var2 = 0
n_intro_var2 = 10

T_INTRO = t_intro_var2
N_INTRO = n_intro_var2

class SIRD:
    def __init__(self, initial_conditions, static_parameters, time_delta):
        self.initial_conditions = initial_conditions
        self.static_parameters = static_parameters
        self.start_time = time_delta[0]
        self.end_time = time_delta[1]
        self.wsol = None
        self.N = None

        # assert(len(self.initial_conditions) == (3 + n_var), f"number of initial conditions does not fit wit@h expected variations, got: {n_var})

        self.initial_conditions_keys = []
        for key,val in self.initial_conditions.items():
            setattr(self, key, val)
            self.initial_conditions_keys.append(key)

        for key,val in self.static_parameters.items():
            setattr(self, key, val)


        # assert self.N is not none to make sure it was set?
        self.calc_end_conditions()

        # self.get_end_condition_dict()


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
        S, I_a, I_b, I_aa, I_bb, I_ba, I_ab, R_a, R_b, R, D = w
        alpha_a, alpha_b, alpha_aa, alpha_bb, alpha_ba, alpha_ab, beta_a, beta_b, gamma_a, gamma_b = p
        t_b = 25
        # Create f = (x1',y1',x2',y2'):
        I_A = I_a + I_aa + I_ba
        I_B = I_b + I_bb + I_ab
        f = [ - (alpha_a/(self.N) * I_A + alpha_b/(self.N) * I_B ) * S,
            (alpha_a/(self.N)) * S * I_A - beta_a * I_a - gamma_a * I_a, # should use all infected?? (I_a + I_aa + I_ba)
            (alpha_b/(self.N)) * S * I_B - beta_b * I_b - gamma_b * I_b,
            (alpha_aa/(self.N)) * R_a * I_A - beta_a * I_aa - gamma_a * I_aa,
            (alpha_bb/(self.N)) * R_b * I_B - beta_b * I_bb - gamma_b * I_bb,
            (alpha_ba/(self.N)) * R_b * I_A - beta_a * I_ba - gamma_a * I_ba,
            (alpha_ab/(self.N)) * R_a * I_B - beta_b * I_ab - gamma_b * I_ab,
            beta_a * (I_a) - (alpha_aa/(self.N)) * R_a * I_A - (alpha_ab/(self.N)) * R_a * I_B,
            beta_b * (I_b) - (alpha_bb/(self.N)) * R_b * I_B - (alpha_ba/(self.N)) * R_b * I_A,
            beta_a * (I_aa + I_ba) + beta_b * (I_bb + I_ab),
            gamma_a * I_A + gamma_b * I_B,
            ]
        return f

    def calc_end_conditions(self, numpoints=None):
        starttime = self.start_time
        stoptime = self.end_time

        if numpoints is None:
            numpoints = stoptime - starttime

        # ODE solver parametersq
        abserr = 1.0e-8
        relerr = 1.0e-6
        # stoptime = stoptime # in days
        # numpoints = numpoints # 1 point per day

        # Create the time samples for the output of the ODE solver.
        # I use a large number of points, only because I want to make
        # a plot of the solution that looks nice.
        # t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]
        self.t = np.linspace(starttime, stoptime, numpoints+1, endpoint=True)
        # print(t)
        # Pack up the parameters and initial conditions:

        
        p = [val for _, val in self.static_parameters.items()]
        w0 = [val for key, val in self.initial_conditions.items() if key != "N" ]

        # p = [alpha_a, alpha_b, beta_a, beta_a, gamma_a, gamma_b]
        # w0 = [S, I_a, I_b, R, D]

        # Call the ODE solver.
        self.wsol = odeint(self.vectorfield, w0, self.t, args=(p,),
                    atol=abserr, rtol=relerr)

        self.end_conditions_vals = self.wsol[-1, :]
        return self.wsol
    
    def get_end_condition_dict(self):
        self.end_conditions = dict(zip(self.initial_conditions_keys, np.r_[self.N,self.end_conditions_vals]))
        return self.end_conditions

    def get_wsol(self):
        # if self.wsol == None:
        #     self.calc_end_conditions()

        return self.t, self.wsol

    def get_wsol_as_SIRD(self):
        wsol = self.get_wsol()
        S = wsol[:,0]
        I = np.sum(wsol[:,1:7], axis=0)
        R = np.sum(wsol[:,8:10], axis=0)
        D = wsol[:,-1]
        sird = np.c_[S,I,R,D]
        return sird




class SIRD2VAR:
    """ Model containing two following SIRD models were the second model introduces a second variant of the infection"""
    def __init__(self, initial_conditions, static_parameters, time_delta, var2_t, var2_n):
        self.initial_conditions = initial_conditions
        self.static_parameters = static_parameters
        self.var2_infection_time = var2_t
        self.var2_initial_cases = var2_n
        self.start_time = time_delta[0]
        self.end_time = time_delta[1]
        time_delta1 = [self.start_time, self.var2_infection_time]
        time_delta2 = [self.var2_infection_time, self.end_time]

        self.initial_conditions_keys = []
        for key,val in self.initial_conditions.items():
            setattr(self, key, val)
            self.initial_conditions_keys.append(key)

        # self.t = np.linspace(self.start_time, self.end_time, endpoint=True)
        # self.t = np.arange(self.start_time, self.end_time)

        # setup the 2 models and carry over the end condition from the first model, set the amount of cases for the new variant.
        self.sird1 = SIRD(initial_conditions=initial_conditions, static_parameters=static_parameters, time_delta=time_delta1)
        initial_conditions_2 = self.sird1.get_end_condition_dict()
        initial_conditions_2["I_b"] += self.var2_initial_cases
        initial_conditions_2["S"] -= self.var2_initial_cases
        self.sird2 = SIRD(initial_conditions=initial_conditions_2, static_parameters=static_parameters, time_delta=time_delta2)

        self.t = np.hstack([self.sird1.t,self.sird2.t])

        self.get_wsol()

    def get_wsol(self):
        combined_model = np.vstack([self.sird1.get_wsol()[1], self.sird2.get_wsol()[1]])
        self.wsol = combined_model
        
        return self.t, combined_model

    def get_wsol_as_SIRD(self):
        _, wsol = self.get_wsol()
        S = wsol[:,0]
        I = np.sum(wsol[:,1:6], axis=1)
        R = np.sum(wsol[:,7:10], axis=1)
        D = wsol[:,-1]
        sird = np.c_[S,I,R,D]
        return self.t, sird

    def plot_wsol(self):
        plot_legend = [key for key, vel in self.initial_conditions.items() if key != "N"]
        plt.figure()
        plt.plot(self.t, self.wsol[:,[0]], 'b-', lw=1) #susciptible
        plt.gca().set_prop_cycle(None)
        plt.plot(self.t, self.wsol[:,[1, 2]], '-') # first infections
        plt.gca().set_prop_cycle(None)
        plt.plot(self.t, self.wsol[:,[3, 4]], '--') # second infections
        # plt.plot(self.t, self.wsol, '--') # second infections
        plt.gca().set_prop_cycle(None)
        plt.plot(self.t, self.wsol[:,[5, 6]], ':') # second cross-infections
        plt.gca().set_prop_cycle(None)
        plt.plot(self.t, self.wsol[:,[7, 8]], '-.', lw=0.7) # first recovery
        # plt.plot(self.t, self.wsol[:,[9]], 'g-', lw=1) # second recovery
        plt.plot(self.t, self.wsol[:,[10]], 'r-', lw=1) # dead
        plt.vlines(self.var2_infection_time, ymin=0, ymax=1e5, lw= 0.5, ls = '--', color='k')
        # plt.hlines(0, 0, 1000)
        plt.legend(plot_legend)
    
    def plot_sird(self):
        t, sird = self.get_wsol_as_SIRD()
        
        plt.figure()
        plt.plot(t, sird[:,:])
        plt.legend(['S', 'I', 'R', 'D'])
        # plt.show()

    @staticmethod
    def solve_by_params( alpha_a, alpha_b, alpha_aa, alpha_bb, alpha_ba, alpha_ab, beta_a, beta_b, gamma_a, gamma_b, t_intro=T_INTRO, n_intro=N_INTRO):
   
        static_params_temp = {
            "alpha_a": alpha_a, # from S to I_a to R_a
            "alpha_b": alpha_b, # from S to I_b to R_b
            "alpha_aa": alpha_aa, # from R_a to I_a to R, reinfection with a from a
            "alpha_bb": alpha_bb, # from R_b to I_b to R , reinfection with b from b recovered
            "alpha_ba": alpha_ba, # from R_b to I_a to R , reinfection with a from b recovered
            "alpha_ab": alpha_ab, # from R_a to I_b to R , reinfection with b from a recovered
            "beta_a": beta_a, # recovery from I_a
            "beta_b": beta_b, # recovery from I_b
            "gamma_a": gamma_a, # death from a
            "gamma_b": gamma_b, # death from b
        }

        init_condtions_temp = {
            "N": S,
            "S": S,
            "I_a": I_a,
            "I_b": 0,
            "I_aa": 0,
            "I_bb": 0,
            "I_ba": 0,
            "I_ab": 0,
            "R_a": 0,
            "R_b": 0,
            "R": R,
            "D": D,
        }

        sird2var_model_temp = SIRD2VAR(initial_conditions=init_condtions_temp, static_parameters=static_params_temp, time_delta=[0, 365*2], var2_t=t_intro, var2_n=1)

        # t, wsol = sird2var_model_temp.get_wsol_as_SIRD()
        t, wsol = sird2var_model_temp.get_wsol()

        return t, wsol, len(t)

    @staticmethod
    def solve_by_params_simple( alpha_a, alpha_b, beta_a, beta_b, gamma_a, gamma_b, t_intro=T_INTRO, n_intro=N_INTRO):

        static_params_temp = {
            "alpha_a": alpha_a, # from S to I_a to R_a
            "alpha_b": alpha_b, # from S to I_b to R_b
            "alpha_aa": alpha_a*0.0, # from R_a to I_a to R, reinfection with a from a
            "alpha_bb": alpha_b*0.0, # from R_b to I_b to R , reinfection with b from b recovered
            "alpha_ba": alpha_a*0.0, # from R_b to I_a to R , reinfection with a from b recovered
            "alpha_ab": alpha_b*0.0, # from R_a to I_b to R , reinfection with b from a recovered
            "beta_a": beta_a, # recovery from I_a
            "beta_b": beta_b, # recovery from I_b
            "gamma_a": gamma_a, # death from a
            "gamma_b": gamma_b, # death from b
        }

        init_condtions_temp = {
            "N": S,
            "S": S,
            "I_a": I_a,
            "I_b": 0,
            "I_aa": 0,
            "I_bb": 0,
            "I_ba": 0,
            "I_ab": 0,
            "R_a": 0,
            "R_b": 0,
            "R": R,
            "D": D,
        }

        sird2var_model_temp = SIRD2VAR(initial_conditions=init_condtions_temp, static_parameters=static_params_temp, time_delta=[0, 365*2], var2_t=t_intro, var2_n=1)

        # t, wsol = sird2var_model_temp.get_wsol_as_SIRD()
        t, wsol = sird2var_model_temp.get_wsol()

        return t, wsol, len(t)


    @staticmethod
    def solve_by_params_alpha_tau( alpha_a, alpha_b, tau_a, tau_b, gamma_a, gamma_b, t_intro=T_INTRO, n_intro=N_INTRO):

        static_params_temp = {
            "alpha_a": alpha_a/tau_a, # from S to I_a to R_a
            "alpha_b": alpha_b/tau_b, # from S to I_b to R_b
            "alpha_aa": alpha_a*0.0, # from R_a to I_a to R, reinfection with a from a
            "alpha_bb": alpha_b*0.0, # from R_b to I_b to R , reinfection with b from b recovered
            "alpha_ba": alpha_a*0.0, # from R_b to I_a to R , reinfection with a from b recovered
            "alpha_ab": alpha_b*0.0, # from R_a to I_b to R , reinfection with b from a recovered
            "beta_a": beta_a/(tau_a**2), # recovery from I_a
            "beta_b": beta_b/(tau_b**2), # recovery from I_b
            "gamma_a": gamma_a, # death from a
            "gamma_b": gamma_b, # death from b
        }

        init_condtions_temp = {
            "N": S,
            "S": S,
            "I_a": I_a,
            "I_b": 0,
            "I_aa": 0,
            "I_bb": 0,
            "I_ba": 0,
            "I_ab": 0,
            "R_a": 0,
            "R_b": 0,
            "R": R,
            "D": D,
        }

        sird2var_model_temp = SIRD2VAR(initial_conditions=init_condtions_temp, static_parameters=static_params_temp, time_delta=[0, 365*2], var2_t=t_intro, var2_n=n_intro)

        # t, wsol = sird2var_model_temp.get_wsol_as_SIRD()
        t, wsol = sird2var_model_temp.get_wsol()

        return t, wsol, len(t)

        


    """
    reinfection matrix []

    """
        
""" Default parameters used if none else declared"""


## DEFINE CROSS PARAMS IN MODEL TO BE USED GENERALLY IN SOLVER

# # Parameter values
# # A = 2.0
# beta_a = 0.032 # recovery rate
# alpha_a = 0.07 #0.07 # A * beta_a # infection rate
# print(alpha_a)

# # B = 2.5
# beta_b = 0.03 # recovery rate
# alpha_b =  0.17 #0.18 #B * beta_b # infection rate
# print(alpha_b)

# t_intro_var2 = 200
# n_intro_var2 = 1

tau_a = 1.4
# tau = sqrt(A)
alpha_a = 0.7 # infection rate
beta_a = alpha_a / tau_a # recovery rate
alpha_a /= tau_a
beta_a /= tau_a
print(alpha_a, beta_a)

tau_b = 4
alpha_b = 0.3 # infection rate
beta_b = alpha_b / tau_b # recovery rate
alpha_b /= tau_b
beta_b /= tau_b
print(alpha_b, beta_b)



static_params_1 = {
    "alpha_a": alpha_a, # from S to I_a to R_a
    "alpha_b": alpha_b, # from S to I_b to R_b
    "alpha_aa": alpha_a * 0,# 0.5, # from R_a to I_a to R, reinfection with a from a
    "alpha_bb": alpha_b * 0,#0.5, # from R_b to I_b to R , reinfection with b from b recovered
    "alpha_ba": alpha_a * 0,#0.9, # from R_b to I_a to R , reinfection with a from b recovered
    "alpha_ab": alpha_b * 0,#0.9, # from R_a to I_b to R , reinfection with b from a recovered
    "beta_a": beta_a, # recovery from I_a
    "beta_b": beta_b, # recovery from I_b
    "gamma_a": 0.0005, # death from a
    "gamma_b": 0.0005, # death from b
}

static_params_2 = static_params_1.copy()

# Initial conditions
# x1 and x2 are the initial displacements; y1 and y2 are the initial velocities
N = 6000000
I_a = 10
I_b = 0
R_a = 0
R_b = 0
R = 0
D = 0
S = N - (I_a + I_b + D + R)
T_init = 0

init_condtions_1 = {
    "N": S,
    "S": S,
    "I_a": I_a,
    "I_b": I_b,
    "I_aa": 0,
    "I_bb": 0,
    "I_ba": 0,
    "I_ab": 0,
    "R_a": 0,
    "R_b": 0,
    "R": R,
    "D": D,
}

if __name__ == "__main__":

    sird2var_model = SIRD2VAR(initial_conditions=init_condtions_1, static_parameters=static_params_1, time_delta=[0, 365*2], var2_t=t_intro_var2, var2_n=n_intro_var2)
    
    # sird_model_1 = SIRD(static_parameters=static_params_1, initial_conditions=init_condtions_1,time_delta = [0,t_intro_var2], n_var=2)
    # init_condtions_2 = sird_model_1.end_conditions
    # init_condtions_2["I_b"] += n_intro_var2
    # init_condtions_2["S"] -= n_intro_var2



    # sird_model_2 = SIRD(static_parameters=static_params_2, initial_conditions=init_condtions_2, time_delta = [t_intro_var2,365*2], n_var=2)


    # combined_model = np.vstack([sird_model_1.all_conditions, sird_model_2.all_conditions])
    # combined_time = np.hstack([sird_model_1.t,sird_model_2.t])
    t, wsol = sird2var_model.get_wsol()
    _, sird_cond = sird2var_model.get_wsol_as_SIRD()

    # plot_legend = [key for key, vel in sird2var_model.initial_conditions.items() if not (key is "N")]



    print("total of each ",(wsol[-1,:]).astype(np.int32) )
    print("total ",np.sum(np.round(wsol[-1,:])) )


    # plt.figure()
    # plt.plot(t, wsol)
    # plt.vlines(t_intro_var2, ymin=0, ymax=N, lw= 0.5, ls = '--', color='k')
    # plt.legend(plot_legend)

    sird2var_model.plot_wsol()
    sird2var_model.plot_sird()

    # plt.figure()
    # plt.plot(t, sird_cond)
    # plt.vlines(t_intro_var2, ymin=0, ymax=N, lw= 0.5, ls = '--', color='k')
    # # plt.plot(combined_time, combined_model)
    # plt.legend(['S', 'I', 'R', 'D'])

    # plt.legend(plot_legend)
    plt.show()