# from SIRD2VAR_class import SIRD, SIRD2VAR
import numpy as np
import torch
from scipy.integrate import odeint
import copy
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons


class SlidersHandle:
    def __init__(self, plot_callback, parameters_dict, fig, ax, lines):
        self.sliders = []
        self.plot_callback = plot_callback
        self.fig = fig
        for i, (param, val) in enumerate(parameters_dict.items()):
            slider_ax  = fig.add_axes([0.1, 0.02 + i*0.03, 0.80, 0.03], facecolor="white")
            slider = Slider(slider_ax, param, 0.000, val*5, valinit=val)
            slider.on_changed(self.sliders_on_changed)
            self.sliders.append(slider)

        # self.lines = ax.plot(*plot_callback(*tuple(parameters_dict.values())), linewidth=2)
        self.lines = lines

    def sliders_on_changed(self, val):
        for line, data in zip(self.lines, self.plot_callback(*self.sliders_val())[1].T):
            line.set_ydata(data)
        # ax.set_ylim([0, N_slider.val])
        # line_data.set_ydata(data)
        self.fig.canvas.draw_idle()
    

    def sliders_val(self):
        slider_vals = [slider.val for slider in self.sliders]
        return tuple(slider_vals)


    def add_slider(self):
        pass


class DiseaseModel():
    def __init__(self, initial_conditions, static_parameters, pde, time_delta=None, description=""):
        self.pde = None
        self.pinn_pde_loss_equations = None
        self.solution = None
        self.solution_sird = None
        self._set_initital_conditions(initial_conditions)
        self._set_static_parameters(static_parameters)
        self._set_time_delta(time_delta)
        self._set_pde_equation(pde)
        self._set_sird_groups()
        self.description = description

        self.ALPHA_STD = 0.15
        self.BETA_STD = 0.1
        self.GAMMA_STD = 0.001
        self.LAMBDA_STD = 2.0
        self.TAU_STD = 10.0
        self.KAPPA_STD = 0.5

        self.S_STD = 1_000_000
        self.I_STD = 1
        self.R_STD = 0
        self.D_STD = 0
        self.IM_STD = 0.0

    def initialize(self, initial_conditions, static_parameters, time_delta=None):
        # assert self.pde != None, "model has no PDE equations set."
        self.set_new_initial_conditions(initial_conditions)
        self.set_new_static_parameters(static_parameters)
        self._set_time_delta(time_delta)
        self._set_pde_equation(self.pde_equations_str)

    def initialize_from_model(self, model_in):
        self.set_new_initial_conditions(model_in.initial_conditions)
        self.set_new_static_parameters(model_in.static_parameters)
        self._set_time_delta(model_in.time_delta)
        self._set_pde_equation(model_in.pde_equations_str)


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
        # self.final_conditions_vals = self.solution[-1, :]
        # self.final_conditions = np.r_[self.N, self.final_conditions_vals]
        # self.final_conditions_dict = dict(zip(['N']+self.initial_conditions_keys,  self.final_conditions))
        self.final_conditions_dict = dict(zip(self.initial_conditions_keys,  self.final_conditions_vals))
        
    def get_final_condition_dict(self):
        self._set_final_conditions()
        return self.final_conditions_dict

    def _set_pde_equation(self, pde_eq_as_str_list):
        self.pde_equations_str = pde_eq_as_str_list
        self._set_pinn_pde_equation()
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
        else:
            print("Could not realise pde as some values are nan.")

    def _set_pinn_pde_equation(self):
        self.pinn_pde_loss_equations = []
        for i, equation in enumerate( self.pde_equations_str):
            pinn_pde_loss_equation = f"d{self.initial_conditions_keys[i]}_t - ({equation})" # subtract entire equation from calculated value
            self.pinn_pde_loss_equations.append(pinn_pde_loss_equation)

    def simulate(self, time_delta=None, numpoints=None):
        # assert self.pde != None , "PDE has not been setup!"
        # assert any(~np.isnan(self.static_parameters_values)), "Static parameters have not been setup! Do this with model.initialize(initial_conditons, static_parameters)"
        # assert any(~np.isnan(self.initial_conditions_values)), "Initial conditions have not been setup! Do this with initialize(initial_conditons, static_parameters)"
        assert self.is_initialized() , "Initial conditions and/or static parameters have not been setup! Do this with model.initialize(initial_conditons, static_parameters)"

        if time_delta is None:
            starttime = self.start_time
            stoptime = self.end_time
        else:
            starttime = time_delta[0]
            stoptime = time_delta[1]

        # stoptime = stoptime # in days
        if numpoints is None:
            numpoints = stoptime - starttime
            self.t = np.linspace(starttime, stoptime, numpoints, endpoint=False)
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

        if len(self.solution) == 0 :
            self.final_conditions_vals = self.initial_conditions_values
        else:
            self.final_conditions_vals = self.solution[-1, :]

        self._set_final_conditions()

        return self.t, self.solution

    # def _induce_infected(self, amount):
    #     pass

    # TODO def return solution as SIRD or SIR, use initial letter in each key
    def _set_sird_groups(self):
        # if self.solution is None:
        #     self.simulate()
        
        black_list_groups = ["Im", "Im_a", "Im_b"] # guesses of groups not in sird
        sird_groups_strings = np.array(['S', 'I', 'R', 'D'])

        n = 0
        for  key_sird in sird_groups_strings:
            for key_model in self.initial_conditions_keys:
                if key_sird in key_model:
                    n += 1
                    break
            # if key_sird in key_model:
            #     continue

        sird_groups_strings = sird_groups_strings[:n]
        self.sird_groups = list(sird_groups_strings)

        # self.solution_sird = np.zeros((self.solution.shape[0], n))
        self.sol_idx = [[] for i in range(n+1)]
        self.sird_idx = []
        self.black_list_groups = [] # actual groups of the model not in sird!

        for i, key in enumerate(self.initial_conditions_keys):
            if any(key[0] == sird_groups_strings) and not (key in black_list_groups):
                grp_idx = np.argwhere(key[0] == sird_groups_strings).flatten()
                # self.solution_sird[:, grp_idx] += np.atleast_2d(self.solution[:, [i]])
                self.sol_idx[grp_idx.item()].append(i)
                self.sird_idx.append(grp_idx.item())
            else:
                grp_idx = n
                self.sol_idx[grp_idx].append(i)
                self.sird_idx.append(grp_idx)
                self.black_list_groups.append(key)

        self.n_sird_groups = n



    def get_solution_as_sird(self):
        if not (self.solution_sird is None):
            return self.t, self.solution_sird
        if self.solution is None:
            self.simulate()
        
        self.solution_sird = np.zeros((self.solution.shape[0], self.n_sird_groups))

        for i, grp_idx in enumerate(self.sird_idx):
            if grp_idx >= self.n_sird_groups:
                continue
            self.solution_sird[:, [grp_idx]] += self.solution[:, [i]]

        return self.t, self.solution_sird

    def plot_solution(self):
        plt.figure()
        for i, key in enumerate(self.initial_conditions_keys):
            ls = '-'
            color = f"C{i}"
            if "Im" in key:
                continue
            try:
                if "S" in key:
                    # color = color = f"C{0}"
                    pass
                elif "I" in key:
                    ls = '--'
                    # color = color = f"C{1}"
                elif "R" in key:
                    ls = '-.'
                    # color = color = f"C{i}"
                elif "D" in key:
                    color = 'r'
                    # color = color = f"C{i}"
                plt.plot(self.t, self.solution[:,i], label=key, ls=ls, color=color)
            except:
                pass
        plt.legend(self.initial_conditions_keys)

    def plot_sird(self):
        if self.solution_sird is None:
            self.get_solution_as_sird()

        plt.figure()
        for i, grp in enumerate(self.sird_groups):
            try:
                plt.plot(self.t, self.solution_sird[:,i], color=f"C{i}", label=grp)
            except:
                pass
        plt.legend()


    def plot_with_sliders(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0.1, bottom=0.07 + len(self.static_parameters_keys)*0.03, right=0.95, top=0.95)
        ax.set_xlim([0, 700])
        ax.set_ylim([0, 1e6])

        time_delta=[0,700]
        # setattr(self, "plot_model",  self())
        self.plot_model = object.__new__(self.__class__)
        self.plot_model.__init__(time_delta=time_delta)

        def plot_callback(*args):
            initial_conditions, static_parameters = self.get_standard_parameters()
            if not (not args):
                for arg, (key, val) in zip(args, static_parameters.items()):
                    static_parameters[key] = arg

            self.plot_model.initialize(initial_conditions, static_parameters, time_delta)
            model = self.plot_model.simulate()
            # print(np.sum(model[1], axis=1))
            return model

        lines = ax.plot(*plot_callback(), linewidth=2)
        ax.legend(self.initial_conditions_keys)

        sliders_handle = SlidersHandle(plot_callback, self.plot_model.static_parameters, fig, ax, lines)
        plt.show()

    def is_initialized(self):
        params_initialized =  any(~np.isnan(self.static_parameters_values))
        conditions_initalized =  any(~np.isnan(self.initial_conditions_values))
        pde_initialized = ~(self.pde == None)
        return params_initialized and conditions_initalized and pde_initialized

    def add_solver(self, solver):
        self.solver = solver(self)

    def get_standard_parameters(self):
        std_static_parameters = {}
        for static_param in self.static_parameters_keys:
            if "alpha" in static_param:
                std_val = self.ALPHA_STD
            elif "beta" in static_param:
                std_val = self.BETA_STD
            elif "gamma" in static_param:
                std_val = self.GAMMA_STD
            elif "lambda" in static_param:
                std_val = self.LAMBDA_STD
            elif "tau" in static_param:
                std_val = self.TAU_STD
            elif "kappa" in static_param:
                std_val = self.KAPPA_STD
            std_static_parameters[static_param] = std_val

        std_initial_conditions = {}
        for cond in self.initial_conditions_keys:
            if "S" in cond:
                std_val = self.S_STD
            elif "I" in cond and not "Im" in cond:
                std_val = self.I_STD
            elif "R" in cond:
                std_val = self.R_STD
            elif "D" in cond:
                std_val = self.D_STD
            elif "Im" in cond:
                std_val = self.IM_STD
            std_initial_conditions[cond] = std_val

        return std_initial_conditions, std_static_parameters






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
        pinn_equations = "\n\t".join(self.pinn_pde_loss_equations)
        print_string += f"PINN PDE loss equations:\n\t{pinn_equations}"
        # TODO add time to print
        return print_string


class DualDiseaseModel(DiseaseModel):
    def __init__(self, initial_conditions, static_parameters, pde, time_delta=None, description="", var2_n_introduced_infected=None):
        self.var2_n_introduced_infected = var2_n_introduced_infected

        self.var1 = DiseaseModel(initial_conditions, static_parameters, pde, time_delta, description)
        self.var2 = DiseaseModel(initial_conditions, static_parameters, pde, time_delta, description)
        super().__init__(initial_conditions, static_parameters, pde, time_delta, description)

    def simulate(self, numpoints=None, var2_n_introduced_infected=1):
        self.var1 = DiseaseModel(self.initial_conditions, self.static_parameters, self.pde_equations_str, self.time_delta, description="")
        self.var2 = DiseaseModel(self.initial_conditions, self.static_parameters, self.pde_equations_str, self.time_delta, description="")

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
        self.solution = np.vstack([self.var1.solution, self.var2.solution])
        return self.t, self.solution

    def initialize(self, initial_conditions, static_parameters, time_delta=None):
        super().initialize(initial_conditions, static_parameters, time_delta)
        self.var1.initialize(initial_conditions, static_parameters, time_delta)
        self.var2.initialize(initial_conditions, static_parameters, time_delta)


    def _induce_infected(self, amount):
        self.final_conditions_t['S'] -= amount
        self.final_conditions_t['Ib'] += amount

    # def _set_initital_conditions(self, initial_conditions_dict):
    #     super()._set_initital_conditions(initial_conditions_dict)
    #     self.var1._set_initital_conditions(initial_conditions_dict)
    #     self.var2._set_initital_conditions(initial_conditions_dict)

""" Below are predefined models that can easily be imported."""


class SIR(DiseaseModel):
    """ Predefined model of the standard SIR model with Susceptible, Infection, and Removed, and alpha, beta as static parameters"""
    def __init__(self, initial_conditions=None, static_parameters=None, time_delta=None, description=""):
        description = "The standard SIR model with Susceptible, Infection, and Removed, and alpha, beta as static parameters"
        if initial_conditions is None:
            initial_conditions = {
                "S": np.nan,
                "I": np.nan,
                "R": np.nan,
                                }
        if static_parameters is None:
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
        if initial_conditions is None:
            initial_conditions = {
                "S": np.nan,
                "I": np.nan,
                "R": np.nan,
                "D": np.nan,
                }
        if static_parameters is None:
            static_parameters = {
                "alpha": np.nan,
                "beta": np.nan,
                "gamma": np.nan,
                }
        pde = [ 
            "-((alpha)/N)*I*S",
            "((alpha)/N)*S*I - (beta)*I - (gamma)*I",
            "(beta)*I ",
            "(gamma)*I",
            ]
        super().__init__(initial_conditions, static_parameters, pde, time_delta, description)




class SIRDIm(DiseaseModel):
    """ Predefined model with Susceptible, Infection, Recovered,  Dead and Immunity Factor and alpha, beta, gamma, kappa as static parameters"""
    def __init__(self, initial_conditions=None, static_parameters=None, time_delta=None, description=""):
        description = "A model that simulates continous reinfection and natural herd immunity as a factor of the amount of recovered"
        if initial_conditions is None:
            initial_conditions = {
                "S": np.nan,
                "I": np.nan,
                "R": np.nan,
                "D": np.nan,
                "Im": np.nan, # should be between 0 and 1
                }
        if static_parameters is None:
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
            "kappa*beta*I/N" # immunity decay  - Im*0.005
            ]
        super().__init__(initial_conditions, static_parameters, pde, time_delta, description)


class SIRDImRel(DiseaseModel):
    """ Predefined model with Susceptible, Infection, Recovered,  Dead and Immunity Factor and lambda, gamma, kappa as static parameters, where lambda is the relation lambda = alpha/beta, and assuning beta = 10 days -> 0.1"""
    def __init__(self, initial_conditions=None, static_parameters=None, time_delta=None, description=""):
        description = "A model that simulates continous reinfection and natural herd immunity as a factor of the amount of recovered"
        if initial_conditions is None:
            initial_conditions = {
                "S": np.nan,
                "I": np.nan,
                "R": np.nan,
                "D": np.nan,
                "Im": np.nan, # should be between 0 and 1
                }
        if static_parameters is None:
            static_parameters = {
                "lambda": np.nan,
                "gamma": np.nan,
                "kappa": np.nan,
                }
        pde = [ 
            "-(0.1*lambda_/N)*I*(S)",
            "(0.1*lambda_/N)*(S)*I + (0.1*lambda_/N)*(1 - Im)*(R-D)*I - 0.1*I - gamma*I",
            "0.1*I - (0.1*lambda_/N)*(1 - Im)*(R-D)*I ",
            "gamma*I",
            "kappa*0.1*I/N" # immunity decay:  - Im*0.005
            ]
        super().__init__(initial_conditions, static_parameters, pde, time_delta, description)



class SIRImRel(DiseaseModel):
    """ Predefined model with Susceptible, Infection, Recovered and Immunity Factor and lambda, kappa as static parameters, where lambda is the relation lambda = alpha/beta, and assuning beta = 10 days -> 0.1"""
    def __init__(self, initial_conditions=None, static_parameters=None, time_delta=None, description=""):
        description = "A model that simulates continous reinfection and natural herd immunity as a factor of the amount of recovered"
        if initial_conditions is None:
            initial_conditions = {
                "S": np.nan,
                "I": np.nan,
                "R": np.nan,
                "Im": np.nan, # should be between 0 and 1
                }
        if static_parameters is None:
            static_parameters = {
                "lambda": np.nan,
                "kappa": np.nan,
                }
        pde = [ 
            "-(0.1*lambda_/N)*I*(S)",
            "(0.1*lambda_/N)*(S)*I + (0.1*lambda_/N)*(1 - Im)*(R)*I - 0.1*I",
            "0.1*I - (0.1*lambda_/N)*(1 - Im)*(R)*I ",
            "kappa*0.1*I/N" # immunity decay:  - Im*0.005
            ]
        super().__init__(initial_conditions, static_parameters, pde, time_delta, description)


class SIRIm(DiseaseModel):
    """ Predefined model with Susceptible, Infection, Recovered and Immunity Factor and lambda, kappa as static parameters, where lambda is the relation lambda = alpha/beta, and assuning beta = 10 days -> 0.1"""
    def __init__(self, initial_conditions=None, static_parameters=None, time_delta=None, description=""):
        description = "A model that simulates continous reinfection and natural herd immunity as a factor of the amount of recovered"
        if initial_conditions is None:
            initial_conditions = {
                "S": np.nan,
                "I": np.nan,
                "R": np.nan,
                "Im": np.nan, # should be between 0 and 1
                }
        if static_parameters is None:
            static_parameters = {
                "alpha": np.nan,
                "beta": np.nan,
                "kappa": np.nan,
                }
        pde = [ 
            "-(alpha/N)*I*(S)",
            "(alpha/N)*(S)*I + (alpha/N)*(1 - Im)*(R)*I - beta*I",
            "beta*I - (alpha/N)*(1 - Im)*(R)*I ",
            "kappa*beta*I/N" # immunity decay:  - Im*0.005
            ]
        super().__init__(initial_conditions, static_parameters, pde, time_delta, description)




class SIRD2Var(DualDiseaseModel):
    """ Predefined model with Susceptible, Infection, Recovered,  Dead and Immunity Factor and alpha, beta, gamma, kappa as static parameters"""
    def __init__(self, initial_conditions=None, static_parameters=None, time_delta=None, description="", var2_n_introduced_infected=None):
        description = "A model that simulates two concurrent diseases and natural herd immunity as a factor of the amount of recovered for each variant"
        if initial_conditions is None:
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
        if static_parameters is None:
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
                "-(alpha_a/N)*Ia*S - (alpha_b/N)*Ib*S",
                "(alpha_a/N)*S*Ia + (alpha_a/N)*(1 - Im_a)*(Ra + Rb - D)*Ia - beta_a*Ia - gamma_a*Ia",
                "(alpha_b/N)*S*Ib + (alpha_b/N)*(1 - Im_b)*(Ra + Rb - D)*Ib - beta_b*Ib - gamma_b*Ib",
                "beta_a*Ia - (alpha_a/N)*(1 - (Im_a))*(Ra)*(Ia) - (alpha_b/N)*(1 - (Im_b))*(Ra)*(Ib)",
                "beta_b*Ib - (alpha_a/N)*(1 - (Im_a))*(Rb)*(Ia) - (alpha_b/N)*(1 - (Im_b))*(Rb)*(Ib)",
                "gamma_a*Ia + gamma_b*Ib",
                "kappa_a*beta_a*Ia/N",
                "kappa_b*beta_b*Ib/N",
            ]
        super().__init__(initial_conditions, static_parameters, pde, time_delta, description, var2_n_introduced_infected)


class SIRD2VarRelSimple(DualDiseaseModel):
    """ Predefined model with Susceptible, Infection, Recovered,  Dead and Immunity Factor and alpha, beta, gamma, kappa as static parameters"""
    def __init__(self, initial_conditions=None, static_parameters=None, time_delta=None, description="", var2_n_introduced_infected=None):
        description = "A model that simulates two concurrent diseases and natural herd immunity as a factor of the amount of recovered for each variant"
        if initial_conditions is None:
            initial_conditions = {
                "S": np.nan,
                "Ia": np.nan,
                "Ib": np.nan,
                "Ra": np.nan,
                "Rb": np.nan,
                "Im_a": np.nan, # should be between 0 and 1
                "Im_b": np.nan, # should be between 0 and 1
                }
        if static_parameters is None:
            static_parameters = {
                "lambda_a": np.nan,
                "lambda_b": np.nan,
                "kappa_a": np.nan,
                "kappa_b": np.nan,
                }
        pde = [ 
            "-(0.1*lambda_a/N)*Ia*S - (0.1*lambda_b/N)*Ib*S",
            "(0.1*lambda_a/N)*S*Ia + (0.1*lambda_a/N)*(1 - Im_a)*(Ra + Rb)*Ia - 0.1*Ia",
            "(0.1*lambda_b/N)*S*Ib + (0.1*lambda_b/N)*(1 - Im_b)*(Ra + Rb)*Ib - 0.1*Ib ",
            "0.1*Ia - (0.1*lambda_a/N)*(1 - (Im_a))*(Ra)*(Ia) - (0.1*lambda_b/N)*(1 - (Im_b))*(Ra)*(Ib)",
            "0.1*Ib - (0.1*lambda_a/N)*(1 - (Im_a))*(Rb)*(Ia) - (0.1*lambda_b/N)*(1 - (Im_b))*(Rb)*(Ib)",
            "kappa_a*0.1*Ia/N",
            "kappa_b*0.1*Ib/N",
            ]
        super().__init__(initial_conditions, static_parameters, pde, time_delta, description, var2_n_introduced_infected)
    



# class SIRD2Var(DiseaseModel):
#     """ Predefined model with Susceptible, Infection, Recovered,  Dead and Immunity Factor and alpha, beta, gamma, kappa as static parameters"""
#     def __init__(self, initial_conditions=None, static_parameters=None, time_delta=None, description="", var2_n_introduced_infected=None):
#         description = "A model that simulates two concurrent diseases and natural herd immunity as a factor of the amount of recovered for each variant"
#         if initial_conditions is None:
#             initial_conditions = {
#                 "S": np.nan,
#                 "Ia": np.nan,
#                 "Ib": np.nan,
#                 "Ra": np.nan,
#                 "Rb": np.nan,
#                 "D": np.nan,
#                 "Im_a": np.nan, # should be between 0 and 1
#                 "Im_b": np.nan, # should be between 0 and 1
#                 }
#         if static_parameters is None:
#             static_parameters = {
#                 "alpha_a": np.nan,
#                 "alpha_b": np.nan,
#                 "beta_a": np.nan,
#                 "beta_b": np.nan,
#                 "gamma_a": np.nan,
#                 "gamma_b": np.nan,
#                 "kappa_a": np.nan,
#                 "kappa_b": np.nan,
#                 }
#         pde = [ 
#             "-(alpha_a/N)*Ia*S  -(alpha_b/N)*Ib*S",
#             "(alpha_a/N)*S*Ia + (alpha_a/N)*(1 - Im_a)*(Ra + Rb - D)*Ia - beta_a*Ia - gamma_a*Ia",
#             "(alpha_b/N)*S*Ib + (alpha_b/N)*(1 - Im_b)*(Ra + Rb - D)*Ib - beta_b*Ib - gamma_b*Ib",
#             "beta_a*Ia - (alpha_a/N)*(1 - (Im_a))*(Ra)*(Ia) - (alpha_b/N)*(1 - (Im_b))*(Ra)*(Ib)",
#             "beta_b*Ib - (alpha_a/N)*(1 - (Im_a))*(Rb)*(Ia) - (alpha_b/N)*(1 - (Im_b))*(Rb)*(Ib)",
#             "gamma_a*Ia + gamma_b*Ib",
#             "kappa_a*beta_a*Ia/N",
#             "kappa_b*beta_b*Ib/N",
#             ]

#         self.var2_n_introduced_infected = var2_n_introduced_infected

#         self.var1 = DiseaseModel(initial_conditions, static_parameters, pde, time_delta, description)
#         self.var2 = DiseaseModel(initial_conditions, static_parameters, pde, time_delta, description)
        
#         super().__init__(initial_conditions, static_parameters, pde, time_delta, description)

#     # overwrite the standard simulate function
#     def simulate(self, numpoints=None, var2_n_introduced_infected=1):
#         self.var1 = DiseaseModel(self.initial_conditions, self.static_parameters, self.pde_equations_str, self.time_delta, description="")
#         self.var2 = DiseaseModel(self.initial_conditions, self.static_parameters, self.pde_equations_str, self.time_delta, description="")

#         if len(self.time_delta) == 2:
#             s = self.time_delta[0]
#             t = s
#             e = self.time_delta[1]
#         else:
#             s = self.time_delta[0]
#             t = self.time_delta[1]
#             e = self.time_delta[2]

#         self.var1.simulate(time_delta=[s, t], numpoints=numpoints)
#         self.final_conditions_t = self.var1.get_final_condition_dict()
#         if self.var2_n_introduced_infected == None:
#             self.var2_n_introduced_infected = var2_n_introduced_infected
#         self._induce_infected(self.var2_n_introduced_infected)
#         self.var2.set_new_initial_conditions(self.final_conditions_t)
#         self.var2.simulate(time_delta=[t, e], numpoints=numpoints)

#         # calculate combined output

#         self.t = np.hstack([self.var1.t, self.var2.t])
#         combined_model = np.vstack([self.var1.solution, self.var2.solution])
#         self.solution = combined_model
#         return self.t, self.solution

#     def initialize(self, initial_conditions, static_parameters, time_delta=None):
#         super().initialize(initial_conditions, static_parameters, time_delta)
#         self.var1.initialize(initial_conditions, static_parameters, time_delta)
#         self.var2.initialize(initial_conditions, static_parameters, time_delta)


#     def _induce_infected(self, amount):
#         self.final_conditions_t['S'] -= amount
#         self.final_conditions_t['Ib'] += amount
        

    


class GeneralModelSolver:
    "Sets up a dummy of the model and simulates it by the given parameters"
    def __init__(self, disease_model: DiseaseModel):
        self.disease_model = disease_model
        self.dummy_model = copy.deepcopy(disease_model)

        
    def __call__(self, *args,  **kwargs):
        # TODO check that params and conditons fit the diseaseModel
        # TODO use initial conditions of supplied model instance
        if not self.dummy_model.is_initialized():
            self.initialize_dummy()


        if not kwargs:
            # print(len(args), len(self.disease_model.static_parameters_keys))
            assert len(args) == len(self.disease_model.static_parameters_keys), f"Number of input parameters does not fit with model, should be {len(self.disease_model.static_parameters_keys)}."
            kwargs = dict(zip(self.disease_model.static_parameters_keys, args))
        else:
            set1 = set(kwargs.keys())
            set2 = set(self.disease_model.static_parameters_keys)
            # print(set1 == set2)
            assert set1 == set2, f"Parameters entered to solver does not fit with given model. Parameters are {self.disease_model.static_parameters_keys}"


        self.dummy_model.set_new_static_parameters(kwargs)

        t, sol = self.dummy_model.simulate()
        _, sol_sird = self.dummy_model.get_solution_as_sird()
        return t, sol, sol_sird

    def get_solution_as_sird(self):
        return self.dummy_model.get_solution_as_sird()


    def initialize_dummy(self):
        assert self.disease_model.is_initialized(), "The parent model is not initialized and the solver can therefore not realize the given parameters."
        self.dummy_model.initialize_from_model(self.disease_model)

    


if __name__ == "__main__":
    "Test functions by creating simple SIR model and testing it with GeneralSolver"
    import matplotlib.pyplot as plt

    pde = [ "-(alpha/N) * I*S",
            "(alpha/N)*S*I - beta*I - gamma*I",
            "beta*I",
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

    t, sol, _ = solver(0.066, 0.021) # solver class used here
    t, sol, _ = solver(alpha=0.066, beta=0.021) # solver class used here

    plt.plot(t_synth, sol_synth)
    plt.gca().set_prop_cycle(None)
    plt.plot(t, sol, '--')
    plt.show()
