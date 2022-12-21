import deepxde as dde
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from Plot import Plot





class SIRD_deepxde_net:
    def __init__(self, t, wsol, disease_model,
                 with_neumann=False,
                 model_name="",
                 with_softadapt=None,
                 use_ln_space=False,
                 use_ic_loss=True,
                 use_observe_loss=True,
                 use_observe_I_loss=False,
                 use_observe_sum_loss=False,
                 use_sign_loss=False,
                 use_smooth_loss=False,
                 use_initL1_loss=False,
                 use_infectedL1_loss=False,
                ):
        self.disease_model = disease_model
        self.model_name = model_name

        self.t, self.wsol = t.reshape(len(t), 1), wsol
        # S_sol, I_sol, R_sol, D_sol = wsol[:,0], wsol[:,1], wsol[:,2], wsol[:,3]
        SOL = wsol
        init_num_people = np.sum(wsol[0,:]) # sum up the first row for the total
        self.N = 1 #init_num_people
        self.init_num_people = init_num_people
        # S_sol, I_sol, R_sol, D_sol= S_sol/init_num_people, I_sol/init_num_people,  R_sol/init_num_people, D_sol/init_num_people
        SOL = SOL/init_num_people
        len_sol = SOL.shape[0]
        
        n_groups = len(self.disease_model.initial_conditions_keys) - len(self.disease_model.black_list_groups)
        n_sird_groups = len(self.disease_model.sird_groups)

        
        if with_softadapt is None or with_softadapt is False:
            self.with_softadapt = None
        else:
            self.with_softadapt = True

        self.use_ln_space = use_ln_space

        self.variables_dict = self.disease_model.static_parameters

        timedomain = dde.geometry.TimeDomain(0, max(t)) # NORMALIZE TIME ?

        self.variables = []
        for key in self.disease_model.static_parameters.keys():
            # guess_noise = np.random.normal(0 , 0.05, 1).item()
            if "alpha" in key:
                init_guess = 0.15
            elif "beta" in key:
                init_guess = 0.1
            elif "gamma" in key:
                init_guess = 0.001
            elif "lambda" in key:
                init_guess = 2.0 
            elif "tau" in key:
                init_guess = 10.0
            elif "kappa" in key:
                init_guess = 0.9
            elif "n" in key:
                init_guess = 1e-6
            else:
                init_guess = 0.5
            if self.use_ln_space:
                setattr(self, key, dde.Variable(np.log(init_guess))) # take log of init guess
            else:
                setattr(self, key, dde.Variable(init_guess)) # take log of init guess
            self.variables.append(eval("self." + key)) # variables are automaticly assigned by the dict names

        self.N_pde_eq = len(self.disease_model.pinn_pde_loss_equations)
        self.PDE_names = []  
        for pde_eq in self.disease_model.pinn_pde_loss_equations:
            name = pde_eq.split("-")[0]
            # self.PDE_names.append(f"PDE{i+1}")
            self.PDE_names.append(name)
        
        def pde(t, y):
            for i, key in enumerate(self.disease_model.initial_conditions_keys):
                value_string = key +"="+ f"y[:, {i}:{i+1}]"
                jacobian_string = f"d{key}_t" +"="+ f"dde.grad.jacobian(y, t, i={i})"
                exec(value_string)
                exec(jacobian_string)
            
            for i, key in enumerate(self.disease_model.static_parameters_keys):
                if self.use_ln_space:
                    value_string = key +"="+ f"torch.exp(self.{key})"
                else:
                    value_string = key +"="+ f"self.{key}"
                exec(value_string)

            # if hasattr(self, "n"):
            #         exec("n *= self.init_num_people")

            N = self.N

            pde_loss = []
            for loss_eq in self.disease_model.pinn_pde_loss_equations:
                pde_loss.append(eval(loss_eq)) # NOTE testing squaring the pde-loss to put more emphasis here

            return pde_loss

                # Test points
        # TODO - how do we weight right points higher than earlier points?
        select_points_after_bool = self.t>=0
        t_later = self.t[select_points_after_bool]
        t_later = t_later.reshape(len(t_later),1)
        SOL_later = SOL[np.tile(select_points_after_bool, (1,len(self.disease_model.sird_groups)))].reshape(-1, len(self.disease_model.sird_groups) )

        select_points_before_bool = self.t<=50
        SOL_early = SOL[np.tile(select_points_before_bool, (1,len(self.disease_model.sird_groups)))].reshape(-1, len(self.disease_model.sird_groups) )
        t_early = self.t[select_points_before_bool]
        t_early = t_early.reshape(len(t_early),1)
        
        def boundary(t_inp, on_initial):
            return on_initial and np.isclose(t_inp[0], t[0])
        
        def boundary_right(t_inp, on_final):
            # print(t[-1])
            return on_final and np.isclose(t_inp[0], t[-1])

        """ LOSS CONDTITIONS """

        self.loss_points_dict = {}
        
        # Initial conditions ---------------------
        if use_ic_loss: ## doesn't 100p correctly
            for i, cond in enumerate(self.disease_model.initial_conditions_keys):
                loss_name = f"ic_{cond}"
                if cond == 'S':
                # if not ("Im" in cond): 
                    # self.loss_points_dict[loss_name] = dde.icbc.IC(timedomain, lambda X: torch.tensor(init_num_people).reshape(1,1), boundary, component=i) # THIS DOES NOT WORK RIGHT!
                    continue
                elif "Im" in cond:
                    ic_sol_value = 0.0 # immunities always start at zero
                # try:
                else:
                    ic_sol_value = SOL[0, self.disease_model.sird_idx[i]] 
                # except:
                #     ic_sol_value = SOL[0, i]
                self.loss_points_dict[loss_name] = dde.icbc.IC(timedomain, lambda X: torch.tensor(ic_sol_value).reshape(1,1), boundary, component=i)

            ## TODO Needs to be sums of components
            self.loss_points_dict['ic_S'] = dde.icbc.IC(timedomain, lambda X: torch.tensor(SOL[0,0]).reshape(1,1), boundary, component=0) # this works however, why?!
        
        



        #  observe direct domain/ curve fitting
        # add S - R ≈ I ?
        if use_observe_loss or use_observe_I_loss:
            for i, cond in enumerate(self.disease_model.sird_groups):
                if not ("I" in cond) and use_observe_I_loss:
                    continue
                loss_name = f"observe_{cond}"
                self.loss_points_dict[loss_name] = dde.icbc.PointSetBC(t_later, SOL_later[:,[i]], component=self.disease_model.sol_idx[i])
        if use_observe_sum_loss:
            self.loss_points_dict['observe_SUM'] = dde.icbc.PointSetBC(t_later, np.ones((len_sol, 1)), component=list(range(n_groups))) # sum of the intire population is constant


        # THESE ARE NOT USED AND SHOULD BE REMOVED
        # for i, cond in enumerate(self.disease_model.black_list_groups): # maybe better to do a seperate 
        #     loss_name = f"observe_{cond}"
        #     self.loss_points_dict[loss_name] = dde.icbc.PointSetBC(t_later, (SOL_later[:,[2]]).reshape(len_sol, 1), component=self.disease_model.sol_idx[n_sird_groups][i]) # TODO need a better pseudo model for immunities instead of the recovered curves!
            # self.loss_points_dict[loss_name] = dde.icbc.PointSetBC(t_later, SOL_later[:,[i]], component=self.disease_model.sol_idx[i])
        # self.loss_points_dict['observe_Im_b'] = dde.icbc.PointSetBC(t_later, (SOL_later[:,[2]]/init_num_people).reshape(len_sol, 1), component=max(self.disease_model.sol_idx)+2)

       

        # L1 of component --------------------- to make model simple/sparse
        dummy = t_later, SOL_later[:,[0]]
        if use_infectedL1_loss:
            for i, cond in enumerate(self.disease_model.initial_conditions_keys): # maybe L2 is less penalising
                if cond in ['Ia','Ib']:
                    loss_name = f"L1_norm_full_{cond}" 
                    self.loss_points_dict[loss_name] = dde.icbc.L1NormComponent(*dummy, component=i)


        # penalise negative curves, used for superposition data, essential when predicting hidden curves.
        # Sign of component ---------------------
        if use_sign_loss:
            for i, cond in enumerate(self.disease_model.initial_conditions_keys):
                if cond == 'S':
                    continue
                # elif "Im" in cond:
                #     continue
                loss_name = f"sign_{cond}"
                self.loss_points_dict[loss_name] = dde.icbc.ComponentAbs(*dummy, component=i)

 


        dummy_early = t_early, SOL_early[:,[0]]
        # smoothness of curves -------------
        if use_smooth_loss:
            for i, cond in enumerate(self.disease_model.initial_conditions_keys):
                loss_name = f"smooth_{cond}"
                self.loss_points_dict[loss_name] = dde.icbc.ComponentAbs(*dummy_early, component=i)



        # L1 norm on first part! --------------------
        if use_initL1_loss:
            for i, cond in enumerate(self.disease_model.initial_conditions_keys):
                if 'Im' in cond :
                    loss_name = f"L1_norm_init_{cond}"
                    self.loss_points_dict[loss_name] = dde.icbc.L1NormComponent(*dummy_early, component=i)

        
        # # Final conditions
        # fc_S = dde.DirichletBC(timedomain, lambda X: torch.tensor(S_sol[-1]).reshape(1,1), boundary_right, component=0)
        # fc_I = dde.DirichletBC(timedomain, lambda X: torch.tensor(I_sol[-1]).reshape(1,1), boundary_right, component=1)
        # fc_R = dde.DirichletBC(timedomain, lambda X: torch.tensor(R_sol[-1]).reshape(1,1), boundary_right, component=2)
        # # fc_D = dde.DirichletBC(timedomain, lambda X: torch.tensor(D_sol[-1]).reshape(1,1), boundary_right, component=3)
        
        
        # Neumann
        # TODO fix neumann to  be general with Disease model!
        if with_neumann:
            assert len(self.disease_model.initial_conditions_keys) == n_sird_groups, "Currently this is not good to be used when having hidden curves. Use only with pure SIR or SIRD models!"
            # get the points gradients
            dSOL_dt = np.gradient(SOL, t, axis=0, edge_order=2)
            # TODO make the gradients resistent to noise by maybe taking the average of the 3-5 endpoints
            #initial slope
            for i, cond in enumerate(self.disease_model.initial_conditions_keys):
                loss_name = f"neumann_ic_{cond}"
                self.loss_points_dict[loss_name] = dde.NeumannBC(timedomain, lambda X: torch.tensor(dSOL_dt[0,i]).reshape(1,1), boundary, component=i) # assuming all slopes are very close to zero
            
            # final slope
            
            for i, cond in enumerate(self.disease_model.initial_conditions_keys):
                loss_name = f"neumann_fc_{cond}"
                self.loss_points_dict[loss_name] = dde.NeumannBC(timedomain, lambda X: torch.tensor(dSOL_dt[-1,i]).reshape(1,1), boundary_right, component=i) # assuming all slopes are very close to zero
                

        # self.data = dde.data.PDE(
        self.data = dde.data.TimePDE(
            timedomain,
            pde,
            list(self.loss_points_dict.values()),
            num_domain=30*self.N_pde_eq,
            num_boundary=10,
            anchors=t.reshape(len(t), 1),
        )

    
    def set_synthetic_data(self, t, wsol):
        self.t_synth, self.wsol_synth = t, wsol
    
    def set_nn_synthetic_data(self, t, wsol, wsol_sird):
        self.t_nn_synth, self.wsol_nn_synth, self.wsol_sird_nn_synth = t, wsol, wsol_sird
    
    def init_model(self, activation="tanh", loss="MSE", initializer="Glorot uniform", 
                   lr=0.01, optimizer="adam", print_every=100, nn_layers=2, nn_layer_width=32, loss_weights=None):

        self.hyper_print_every = print_every
        layer_size = [1] + [nn_layer_width] * nn_layers + [self.N_pde_eq] # input is the time, therefore always 1
        
        net = dde.nn.FNN(layer_size, activation, initializer)
        
        self.model = dde.Model(self.data, net)
        
        #TODO - should we add decay here?
        #TODO - batch size (see https://github.com/lululxvi/deepxde/issues/320)
        self.model.compile(optimizer, lr=lr
                        #    ,metrics=["l2 relative error"]
                           ,loss=loss
                           ,external_trainable_variables=self.variables
                           ,loss_weights=loss_weights
                           ,with_softadapt=self.with_softadapt
                      )
        
        self.variable = dde.callbacks.VariableValue(self.variables, period=1, filename=f'variables_{self.model_name}.txt', precision=10)
        #TODO add callback to save the NN and other callbacks?


    def train_model(self, iterations=7500, print_every=1000, use_LBFGSB=False):
        self.print_every = print_every
        self.losshistory, self.train_state = self.model.train(iterations=iterations, 
                                                            callbacks=[self.variable],
                                                            display_every=print_every)

        if use_LBFGSB: ## this is deprecated and is not working
            raise DeprecationWarning("This feature is depcrecated..")
            print("optimizing with L-BFGS-B")
            self.model.compile("L-BFGS-B")
            losshistory, train_state = self.model.train()
            self.best_alpha_nn, self.best_beta_nn = self.variable.get_value() 
        else:
            self._get_best_params()
        self._best_nn_prediction()


    def intermediate_train(self, plot_every=None, reset=False):
        if plot_every == None:
            plot_every = self.hyper_print_every

        TOTAL_ITER = 100_000
        iters = 0
        if not hasattr(self, "prev_best_step") or reset:
            setattr(self, "prev_best_step", 0)
        try:
            while True:
                iters += plot_every
                # for n in range(TOTAL_ITER//plot_every):
                self.train_model(iterations=plot_every, print_every=self.hyper_print_every, use_LBFGSB=False)
                best_step = self.get_best_train_step() # TODO make a get_best_step function
                if best_step > self.prev_best_step:
                    setattr(self, "prev_best_step", best_step)
                    break
                elif iters >= TOTAL_ITER:
                    break
        except KeyboardInterrupt:
            print("Training ended prematurely")

        params_nn = self.get_best_params() 
        print(self.disease_model.static_parameters, sep="\n")
        t_nn_param, wsol_nn_param, wsol_sird_nn_param = self.disease_model.solver(*params_nn)
        # params_nn= tuple(np.exp([*params_nn]))
        # print(*params_nn)
        # self.set_synthetic_data(t_synth, solution_synth_full) 
        self.set_nn_synthetic_data(t_nn_param, wsol_nn_param, wsol_sird_nn_param)
        plot = Plot(self) # class that contains plotting functions
        plot.show_known_and_prediction()


    def _get_param_history(self):
        df = pd.read_csv(f'variables_{self.model_name}.txt', header=None, delimiter=' ', index_col=0)
        self.param_history = df
        return df

    
    def _get_best_params(self):
        df = self._get_param_history()

        for i in range(1, len(self.variables)+1):
            df[i] = (df[i].str[1*(i == 1):-1].astype('float'))

        self.best_params = list(df.loc[self.train_state.best_step])

        for i,((key,val), best_val) in enumerate(zip(self.variables_dict.items(), df)):
            if "n" in key:
                self.best_params[i] *= self.disease_model.number_in_population
            setattr(self, "best_" + key + "_nn", self.best_params[i])

        self.best_params_dict = dict(zip(self.variables_dict.keys(), self.best_params))
        self.best_params_tuple = tuple(self.best_params)

        return df
    
    def get_best_params(self, out_func=lambda x: x):
        best_step = self.get_best_train_step()
        if self.use_ln_space:
            out_func = np.exp
        print(f"Best train step: {best_step}")
        for i,(key, val) in enumerate(self.variables_dict.items()):
            print(f"{key}: {out_func(self.best_params_tuple[i])}" )

        return out_func(self.best_params_tuple)
    
    def get_best_train_step(self):
        return self.model.train_state.best_step

    # TODO def get_params_by_step # including the nn at that step? maybe this is in model.train_state?
    
    def _best_nn_prediction(self):
        y_dim = self.train_state.best_y.shape[1]

        idx = np.argsort(self.train_state.X_test[:, 0])
        self.t_nn_best = self.train_state.X_test[idx, 0]
        n = len(self.disease_model.sird_groups)

        wsol_nn_best = []
        wsol_sird_nn_best = np.zeros((self.train_state.best_y.shape[0], n)) 
        for i in range(y_dim):
            coulmn_to_add = self.train_state.best_y[idx, i]*self.init_num_people
            wsol_nn_best.append(coulmn_to_add)
            sird_idx = self.disease_model.sird_idx[i] 
            if  (sird_idx :=self.disease_model.sird_idx[i] ) < n:
                wsol_sird_nn_best[:,sird_idx] += coulmn_to_add
        self.wsol_nn_best = np.array(wsol_nn_best).T
        self.wsol_sird_nn_best = wsol_sird_nn_best
    
    def get_best_nn_prediction(self):
        return self.t_nn_best, self.wsol_nn_best
    
    def run_all(self, t_synth, wsol_synth, solver, print_every=1000, iterations=8000
                ,layer_size=None, activation="tanh", initializer="Glorot uniform"
                ,lr=0.01, optimizer="adam"):
        self.init_model(print_every=print_every, layer_size=layer_size, activation=activation,
                        initializer=initializer, lr=lr, optimizer=optimizer)
        self.train_model(iterations=iterations, print_every=print_every)
        alpha_nn, beta_nn, gamma_nn = self.get_best_params()
        t_nn_param, wsol_nn_param, N_nn_param = solver.solve_SIRD(alpha_nn, beta_nn, gamma_nn)
        self.set_synthetic_data(t_synth, wsol_synth)
        self.set_nn_synthetic_data(t_nn_param, wsol_nn_param)


    def __str__(self) -> str:
        return "PINN model:" + "\nParameters: " + str(list(self.variables_dict.keys())) + "\nLoss measures: " + str(self.PDE_names + list(self.loss_points_dict.keys()))




if __name__=='__main__':
    import deepxde as dde
    from deepxde.backend import pytorch
    import torch
    import matplotlib.pyplot as plt
    import numpy as np

    from SIRD_deepxde_DiseaseModel import SIRD_deepxde_net
    from DiseaseModel import SIRD2Var, GeneralModelSolver
    from Plot import Plot


    seed = 1
    np.random.seed(seed)
    dde.config.set_random_seed(seed)

    time_delta = [0,3*365] # use three values here for intro time of second variant
    # import ODE_SIRD_reinfection_class as sird2
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
        "alpha_a": 0.11,
        "alpha_b": 0.12,
        "beta_a": 0.08,
        "beta_b": 0.08,
        "gamma_a": 0.00,
        "gamma_b": 0.00,
        "kappa_a": 0.1,
        "kappa_b": 0.2,
        }

    sird_model = SIRD2Var(initial_conditions, static_parameters, time_delta)
    solver = GeneralModelSolver(sird_model)
    t_synth, solution_synth_full = sird_model.simulate()
    t_synth, solution_synth = sird_model.get_solution_as_sird()
    sird_model.plot_solution()
    sird_model.plot_sird()


    t = t_synth
    wsol = solution_synth

    model = SIRD_deepxde_net(t, wsol,disease_model=sird_model, with_neumann=True, model_name="diseasemodel_test", with_softadapt=True, )
    print(model)
    hyper_print_every = 10
    model.init_model(lr=0.01, print_every=hyper_print_every, activation="tanh")


    model.train_model(iterations=100, print_every=hyper_print_every, use_LBFGSB=False)

    params_nn = model.get_best_params(out_func=np.exp) # parameters need to be extracted with the exponential functino as they have been modelled in logspace
    t_nn_param, wsol_nn_param, wsol_sird_nn_param = solver(*params_nn)
    # params_nn= tuple(np.exp([*params_nn]))
    # print(*params_nn)
    model.set_synthetic_data(t_synth, solution_synth_full) 
    model.set_nn_synthetic_data(t_nn_param, wsol_nn_param, wsol_sird_nn_param)
    plot = Plot(model, values_to_plot=sird_model.initial_conditions_keys) # class that contains plotting functions
    plot.show_known_and_prediction()
    plot.plot_param_history()
    plot.plot_loss_history()

    plt.show()










# import deepxde as dde
# import numpy as np
# from deepxde.backend import pytorch
# import torch
# import matplotlib.pyplot as plt
# from SIRD_singlevar_simplified_deepxde_class import SIRD_deepxde_net, Plot
# from SIRD_d import SIRD_deepxde_net, Plot
# from SIRD_deepxde_class import Plot
# from ODE_SIR_copy import ODESolver

    # from ODE_SIRD_singlevar_reinfection_class import SIRD2VAR, static_params_1, init_condtions_1, t_intro_var2, n_intro_var2

    # seed = 1
    # np.random.seed(seed)
    # dde.config.set_random_seed(seed)

    # solver = SIRD2VAR.solve_by_params_alpha_tau

    # # import ODE_SIR
    # # solver = ODE_SIR.ODESolver()
    # # t_synth, wsol_synth, N = solver.solve_SIRD(alpha_real, beta_real, gamma_real)
    # # solver.plot_SIRD(t_synth, wsol_synth)

    # time_delta = [0,2*365]
    # # import ODE_SIRD_reinfection_class as sird2

    # sird_model = SIRD2VAR(init_condtions_1, static_params_1, time_delta, t_intro_var2, n_intro_var2)
    # t_synth, wsol_synth = sird_model.get_wsol_as_SIRD()
    # t_synth, wsol_synth_full = sird_model.get_wsol()
    # sird_model.plot_wsol()
    # sird_model.plot_sird()

    # # keep this even if not subsetting
    # t = t_synth
    # wsol = wsol_synth

    # # subset
    # # max_timestep = 300
    # # t_bool = t_synth < max_timestep
    # # t = t_synth[t_bool]
    # # wsol = wsol_synth[t_bool]

    # # 0.13 0.065 0.08333333333333333
    # variables = {
    #     "alpha_a": 0.05,
    #     "alpha_aa": 0.01,
    #     "beta_a": 0.01
    # }

    # model = SIRD_deepxde_net(t, wsol, with_neumann=False, model_name="singlevar_tau", variables_init_dict=variables, with_softadapt=True)
    # print(model)
    # hyper_print_every = 10
    # model.init_model(lr=0.005, print_every=hyper_print_every)

    # model.train_model(iterations=10, print_every=hyper_print_every, use_LBFGSB=False)


    # # print(f"Best train step: {model.model.train_state.best_step}")
    # params_nn = model.get_best_params()
    # # print('Alpha_a: {}, Alpha_b: {}, Alpha_aa: {}, Alpha_bb: {}, Alpha_ba: {}, Alpha_ab: {}, beta_a: {}, beta_b: {}, gamma_a: {}, gamma_b: {}'.format(*params_nn))
    # # print('Alpha_a: {}, Alpha_b: {}, beta_a: {}, beta_b: {}, gamma_a: {}, gamma_b: {}'.format(*params_nn))
    # # for i,(key, val) in enumerate(variables.items()):
    # #     print(f"{key}: {params_nn[i]}" )
    # # alpha_a_nn, alpha_b_nn, beta_a_nn, beta_b_nn, gamma_a_nn, gamma_b_nn = params_nn
    # t_nn_param, wsol_nn_param, N_nn_param = solver(*params_nn)
    # # plt.plot(t_nn_param, wsol_nn_param)

    # # we need to set the synthetic data as it comes from outside the network
    # # the two functions below sets the synthetic data
    # model.set_synthetic_data(t_synth, wsol_synth_full) 
    # model.set_nn_synthetic_data(t_nn_param, wsol_nn_param)

    # plot = Plot(model, values_to_plot=['S', 'Ia','Iaa', 'Ra', 'R']) # class that contains plotting functions


    # plot.show_known_and_prediction()
    # plot.plot_param_history()
    # plot.plot_loss_history()

    # plt.show()


