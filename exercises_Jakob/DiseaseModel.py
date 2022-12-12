# from SIRD2VAR_class import SIRD, SIRD2VAR
import numpy as np
from scipy.integrate import odeint
import copy


class DiseaseModel():
    def __init__(self, initial_conditions, static_parameters, pde, time_delta=None, description=""):
        self.pde = None
        self.solution = None
        self._set_initital_conditions(initial_conditions)
        self._set_static_parameters(static_parameters)
        self._set_time_delta(time_delta)
        self._set_pde_equation(pde)
        self.description = description

    def initialize(self, initial_conditions, static_parameters, time_delta=None):
        # assert self.pde != None, "model has no PDE equations set."
        self.set_new_initial_conditions(initial_conditions)
        self.set_new_static_parameters(static_parameters)
        self._set_time_delta(time_delta)
        self._set_pde_equation(self.pde_equations_str)


    def _set_time_delta(self, time_delta):
        # TODO if len is 3 middle time is t_intro
        # TODO assert length
        if time_delta == None:
            time_delta = [0, 365]
            # Warning(" Assuming a time delta of a year in unit time of days: time_delta= [0, 365]") # how does this work?
        if len(time_delta) == 3:
            self.time_intro = time_delta[1]
        self.time_delta = time_delta
        self.start_time = self.time_delta[0]
        self.end_time = self.time_delta[-1]


    def _set_static_parameters(self, static_params_dict):
        self.static_parameters = static_params_dict
        self.static_parameters_keys = []
        self.static_parameters_values = []
        
        for key, val in static_params_dict.items():
            self.static_parameters_keys.append(key)
            self.static_parameters_values.append(val)
            setattr(self, key, val)

        self.static_parameters_values = np.array(self.static_parameters_values)

    def set_new_static_parameters(self, new_static_parameters):
        assert set(self.static_parameters_keys) == set(new_static_parameters.keys()), "new parameter names does not fit with the model.."
        self._set_static_parameters(new_static_parameters)

    def set_new_initial_conditions(self, new_initial_condtions):
        assert set(self.initial_conditions_keys) == set(new_initial_condtions.keys()), "new initial condition names does not fit with the model.."
        self._set_initital_conditions(new_initial_condtions)

    def _set_initital_conditions(self, initial_conditions_dict):
        self.initial_conditions = initial_conditions_dict
        self.initial_conditions_keys = []
        self.initial_conditions_values = []

        for key, val in initial_conditions_dict.items():
            # if key != "N":
            self.initial_conditions_keys.append(key)
            self.initial_conditions_values.append(val)
            setattr(self, key, val)
        
        self.initial_conditions_values = np.array(self.initial_conditions_values)
        self.N = np.sum(self.initial_conditions_values) # beware of noise here

    def _set_final_conditions(self):
        self.final_conditions_vals = self.solution[-1, :]
        # self.final_conditions = np.r_[self.N, self.final_conditions_vals]
        # self.final_conditions_dict = dict(zip(['N']+self.initial_conditions_keys,  self.final_conditions))
        self.final_conditions_dict = dict(zip(self.initial_conditions_keys,  self.final_conditions_vals))
        
    def get_final_condition_dict(self):
        self._set_final_conditions()
        return self.final_conditions_dict

    def _set_pde_equation(self, pde_eq_as_str_list):
        self.pde_equations_str = pde_eq_as_str_list
        if all(~np.isnan(self.static_parameters_values)) and all(~np.isnan(self.initial_conditions_values)):
            def vectorfield(w, t, p):
                for (key, w_) in zip(self.initial_conditions_keys , w):
                    string = key +"="+ str(w_)
                    exec(string)
                for (key, p_) in zip(self.static_parameters_keys , p):
                    string = key +"="+ str(p_)
                    exec(string)
                N = self.N
                f = []
                for equation in pde_eq_as_str_list:
                    f.append(eval(equation))
                return f
            setattr(self, "pde", vectorfield)

    def simulate(self, time_delta=None, numpoints=None):
        # assert self.pde != None , "PDE has not been setup!"
        assert any(~np.isnan(self.static_parameters_values)), "Static parameters have not been setup! Do this with model.initialize(initial_conditons, static_parameters)"
        assert any(~np.isnan(self.initial_conditions_values)), "Initial conditions have not been setup! Do this with initialize(initial_conditons, static_parameters)"

        if time_delta is None:
            starttime = self.start_time
            stoptime = self.end_time
        else:
            starttime = time_delta[0]
            stoptime = time_delta[1]

        # stoptime = stoptime # in days
        if numpoints is None:
            numpoints = stoptime - starttime
            self.t = np.linspace(starttime, stoptime, numpoints+1, endpoint=True)
        else:
            self.t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]

        # Pack up the parameters and initial conditions:
        p = self.static_parameters_values
        w0 = [val for key, val in self.initial_conditions.items() if key != "N" ] # have to ignore N

        # ODE solver parametersq
        abserr = 1.0e-8
        relerr = 1.0e-6
        # Call the ODE solver.
        self.solution = odeint(self.pde, w0, self.t, args=(p,),
                    atol=abserr, rtol=relerr)

        self.end_conditions_vals = self.solution[-1, :]
        self._set_final_conditions()

        return self.t, self.solution

    # def _induce_infected(self, amount):
    #     pass

    # TODO def return solution as SIRD or SIR, use initial letter in each key
    # TODO add noise to simulation!
    # TODO make so that a model can be unrealised/ without parameters such that is defined by its pde, done!

    def __str__(self) -> str:
        print_string = f"\nA Disease Model with description: '{self.description}':\n"

        parameters_string = [f"\t{key} = {val}\n" for key, val in self.static_parameters.items()]
        parameters_string = "".join(parameters_string)
        print_string += f"Parameters:\n{parameters_string}\n"

        groups_string = [f"\t{key} = {val}\n" for key, val in self.initial_conditions.items()]
        groups_string = "".join(groups_string)
        print_string += f"PDE groups and initial conditions:\n{groups_string}\n"
        equations_string = [f"\td{g}/dt = {eq}\n" for g, eq in zip(self.initial_conditions_keys, self.pde_equations_str)]
        equations_string = "".join(equations_string)
        print_string += f"PDE equations:\n {equations_string}"
        # TODO add time to print
        return print_string


class SIR(DiseaseModel):
    """ Predefined model of the standard SIR model with Susceptible, Infection, and Removed, and alpha, beta as static parameters"""
    def __init__(self, initial_conditions=None, static_parameters=None, time_delta=None, description=""):
        description = "The standard SIR model with Susceptible, Infection, and Removed, and alpha, beta as static parameters"
        initial_conditions = {
            "S": np.nan,
            "I": np.nan,
            "R": np.nan,
                             }
        static_parameters = {
            "alpha": np.nan,
            "beta": np.nan,
                            }
        pde = [
            "-(alpha/N) * I*S",
            "(alpha/N)*S*I - beta*I",
            "beta*I ",
            ]
        super().__init__(initial_conditions, static_parameters, pde, time_delta, description)


class SIRD(DiseaseModel):
    """ Predefined model of the SIRD model with Susceptible, Infection, Recovered, and Dead, and alpha, beta, gamma as static parameters"""
    def __init__(self, initial_conditions=None, static_parameters=None, time_delta=None, description=""):
        description = "The standard SIRD model with Susceptible, Infection, and Removed, and alpha, beta, gamma as static parameters"
        initial_conditions = {
            "S": np.nan,
            "I": np.nan,
            "R": np.nan,
            "D": np.nan,
            }
        static_parameters = {
            "alpha": np.nan,
            "beta": np.nan,
            "gamma": np.nan,
            }
        pde = [ 
            "-(alpha/N)*I*S",
            "(alpha/N)*S*I - beta*I - gamma*I",
            "beta*I ",
            "gamma*I",
            ]
        super().__init__(initial_conditions, static_parameters, pde, time_delta, description)


class SIRDIm(DiseaseModel):
    """ Predefined model with Susceptible, Infection, Recovered,  Dead and Immunity Factor and alpha, beta, gamma, kappa as static parameters"""
    def __init__(self, initial_conditions=None, static_parameters=None, time_delta=None, description=""):
        description = "A model that simulates continous reinfection and natural herd immunity as a factor of the amount of recovered"
        initial_conditions = {
            "S": np.nan,
            "I": np.nan,
            "R": np.nan,
            "D": np.nan,
            "Im": np.nan, # should be between 0 and 1
            }
        static_parameters = {
            "alpha": np.nan,
            "beta": np.nan,
            "gamma": np.nan,
            "kappa": np.nan,
            }
        pde = [ 
            "-(alpha/N)*I*(S)",
            "(alpha/N)*(S)*I + (alpha/N)*(1 - Im)*(R-D)*I - beta*I - gamma*I",
            "beta*I - (alpha/N)*(1 - Im)*(R-D)*I ",
            "gamma*I",
            "kappa*beta*I/N - Im*0.005"
            ]
        super().__init__(initial_conditions, static_parameters, pde, time_delta, description)


class SIRD2Var(DiseaseModel):
    """ Predefined model with Susceptible, Infection, Recovered,  Dead and Immunity Factor and alpha, beta, gamma, kappa as static parameters"""
    def __init__(self, initial_conditions=None, static_parameters=None, time_delta=None, description="", var2_n_introduced_infected=None):
        description = "A model that simulates two concurrent diseases and natural herd immunity as a factor of the amount of recovered for each variant"
        initial_conditions = {
            "S": np.nan,
            "Ia": np.nan,
            "Ib": np.nan,
            "Ra": np.nan,
            "Rb": np.nan,
            "D": np.nan,
            "Im_a": np.nan, # should be between 0 and 1
            "Im_b": np.nan, # should be between 0 and 1
            }
        static_parameters = {
            "alpha_a": np.nan,
            "alpha_b": np.nan,
            "beta_a": np.nan,
            "beta_b": np.nan,
            "gamma_a": np.nan,
            "gamma_b": np.nan,
            "kappa_a": np.nan,
            "kappa_b": np.nan,
            }
        pde = [ 
            "-(alpha_a/N)*Ia*S  -(alpha_b/N)*Ib*S",
            "(alpha_a/N)*S*Ia + (alpha_a/N)*(1 - Im_a)*(Ra + Rb - D)*Ia - beta_a*Ia - gamma_a*Ia",
            "(alpha_b/N)*S*Ib + (alpha_b/N)*(1 - Im_b)*(Ra + Rb - D)*Ib - beta_b*Ib - gamma_b*Ib",
            "beta_a*Ia - (alpha_a/N)*(1 - (Im_a))*(Ra)*(Ia) - (alpha_b/N)*(1 - (Im_b))*(Ra)*(Ib)",
            "beta_b*Ib - (alpha_a/N)*(1 - (Im_a))*(Rb)*(Ia) - (alpha_b/N)*(1 - (Im_b))*(Rb)*(Ib)",
            "gamma_a*Ia + gamma_b*Ib",
            "kappa_a*beta_a*Ia/N",
            "kappa_b*beta_b*Ib/N",
            ]

        self.var2_n_introduced_infected = var2_n_introduced_infected

        self.var1 = DiseaseModel(initial_conditions, static_parameters, pde, time_delta, description)
        self.var2 = DiseaseModel(initial_conditions, static_parameters, pde, time_delta, description)
        
        super().__init__(initial_conditions, static_parameters, pde, time_delta, description)

    def simulate(self, numpoints=None, var2_n_introduced_infected=1):
        if len(self.time_delta) == 2:
            s = self.time_delta[0]
            t = s
            e = self.time_delta[1]
        else:
            s = self.time_delta[0]
            t = self.time_delta[1]
            e = self.time_delta[2]

        self.var1.simulate(time_delta=[s, t], numpoints=numpoints)
        self.final_conditions_t = self.var1.get_final_condition_dict()
        if self.var2_n_introduced_infected == None:
            self.var2_n_introduced_infected = var2_n_introduced_infected
        self._induce_infected(self.var2_n_introduced_infected)
        self.var2.set_new_initial_conditions(self.final_conditions_t)
        self.var2.simulate(time_delta=[t, e], numpoints=numpoints)

        # calculate combined output

        self.t = np.hstack([self.var1.t, self.var2.t])
        combined_model = np.vstack([self.var1.solution, self.var2.solution])
        self.solution = combined_model
        return self.t, self.solution

    def initialize(self, initial_conditions, static_parameters, time_delta=None):
        super().initialize(initial_conditions, static_parameters, time_delta)
        self.var1.initialize(initial_conditions, static_parameters, time_delta)
        self.var2.initialize(initial_conditions, static_parameters, time_delta)


    def _induce_infected(self, amount):
        self.final_conditions_t['S'] -= amount
        self.final_conditions_t['Ib'] += amount
        

    


class GeneralModelSolver:
    "Sets up a dummy of the model and simulates it by the given parameters"
    def __init__(self, disease_model: DiseaseModel):
        self.disease_model = disease_model

        
    def __call__(self, *args,  **kwargs):
        # TODO check that params and conditons fit the diseaseModel
        # TODO use initial conditions of supplied model instance

        if not kwargs:
            # print(len(args), len(self.disease_model.static_parameters_keys))
            assert len(args) == len(self.disease_model.static_parameters_keys), f"Number of input parameters does not fit with model, should be {len(self.disease_model.static_parameters_keys)}."
            kwargs = dict(zip(self.disease_model.static_parameters_keys, args))
        else:
            set1 = set(kwargs.keys())
            set2 = set(self.disease_model.static_parameters_keys)
            # print(set1 == set2)
            assert set1 == set2, f"Parameters entered to solver does not fit with given model. Parameters are {self.disease_model.static_parameters_keys}"

        dummy_model = copy.deepcopy(self.disease_model)
        dummy_model.set_new_static_parameters(kwargs)

        t, sol = dummy_model.simulate()

        return t, sol

    


if __name__ == "__main__":
    "Test functions by creating simple SIR model and testing it with GeneralSolver"
    import matplotlib.pyplot as plt

    pde = [ "-(alpha/N) * I*S",
            "(alpha/N)*S*I - beta*I - gamma*I",
            "beta*I ",
            "gamma*I",
            ]

    static_params = {"alpha": 0.07,
                     "beta": 0.02,
                    #  "gamma": 0.002
                    }

    init_cond = {"S": 100000,
                 "I": 10,
                 "R": 0,
                #  "D": 0,
                }

    #time_delta

    # SIR = DiseaseModel(init_cond, static_params, pde)
    sir = SIR() # init_conditions and static parameters can also be put in here
    solver = GeneralModelSolver(sir) # uses model to solve by new params
    try:
        sir.simulate()
    except:
        pass
    print(sir)
    sir.initialize(init_cond, static_params, time_delta=[0,1000])
    print(sir)

    t_synth, sol_synth = sir.simulate()

    try:
        t, sol = solver(alpha=0.066, beta=0.021, gamma=0.001) # doesn't accept parameters that does not match the model
    except:
        print("didn't work!")

    t, sol = solver(0.066, 0.021) # solver class used here
    t, sol = solver(alpha=0.066, beta=0.021) # solver class used here

    plt.plot(t_synth, sol_synth)
    plt.gca().set_prop_cycle(None)
    # plt.plot(t, sol, '--')
    plt.show()
