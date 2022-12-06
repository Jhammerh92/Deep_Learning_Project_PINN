import deepxde as dde
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

class Plot:
    def __init__(self, model, colors = None, values_to_plot=['S','I','R','D']):
        self.model = model
        self.values_to_plot = values_to_plot
        
        if colors is None:
            self.colors = ['C0','C1','C2','C3']
    
    def _set_color(self, line, colors):
        ''' Set color for each curve in a figure '''
        for idx, color in enumerate(colors):
            line[idx].set_color(color)
    
    def show_deepxde_plot(self):
        dde.saveplot(self.model.losshistory, self.model.train_state, issave=False, isplot=True)
    
    def show_known_and_prediction(self, figsize=(15,5)):
        
        fig, axes = plt.subplots(1,2
                         #, sharex=True
                         , sharey=True
                         , figsize=figsize)
        
        self.plot_known_data(axes[0])
        
        self.plot_future_prediction(axes[1])
    
    def _plot_known_synthetic(self, ax):
        label_set = False
        s = 1
        if 'S' in self.values_to_plot:
            line = ax.scatter(self.model.t, self.model.wsol[:,0], color=self.colors[0], s=s)
            if not label_set:
                line.set_label('Synthetic')
                label_set = True
        if 'I' in self.values_to_plot:
            line = ax.scatter(self.model.t, self.model.wsol[:,1], color=self.colors[1], s=s)
            if not label_set:
                line.set_label('Synthetic')
                label_set = True
        if 'R' in self.values_to_plot:
            line = ax.scatter(self.model.t, self.model.wsol[:,2], color=self.colors[2], s=s)
            if not label_set:
                line.set_label('Synthetic')
                label_set = True
        if 'D' in self.values_to_plot:
            line = ax.scatter(self.model.t, self.model.wsol[:,3], color=self.colors[3], s=s)
            if not label_set:
                line.set_label('Synthetic')
                label_set = True
        # self._set_color(line, self.colors)
    
    def _plot_known_nn(self, ax, linestyle='--'):
        if 'S' in self.values_to_plot:
            line = ax.plot(self.model.t_nn_best, self.model.wsol_nn_best[:,0], linestyle=linestyle, color=self.colors[0])
        if 'I' in self.values_to_plot:
            line = ax.plot(self.model.t_nn_best, self.model.wsol_nn_best[:,1], linestyle=linestyle, color=self.colors[1])
        if 'R' in self.values_to_plot:
            line = ax.plot(self.model.t_nn_best, self.model.wsol_nn_best[:,2], linestyle=linestyle, color=self.colors[2])
        if 'D' in self.values_to_plot:
            line = ax.plot(self.model.t_nn_best, self.model.wsol_nn_best[:,3], linestyle=linestyle, color=self.colors[3])
            
        line[0].set_label('PINN prediction')
        # self._set_color(line, self.colors)
    
    def plot_known_data(self, ax):
        ax.set_title('Known data')
        
        self._plot_known_synthetic(ax)
        
        self._plot_known_nn(ax)
        
        ax.legend()
    
    def _plot_pred_synthetic(self, ax, **kwargs):
        label_set = False
        if 'S' in self.values_to_plot:
            line = ax.plot(self.model.t_synth, self.model.wsol_synth[:,0], color=self.colors[0], **kwargs)
            if not label_set:
                line[0].set_label('ODE solution')
                label_set = True
        if 'I' in self.values_to_plot:
            line = ax.plot(self.model.t_synth, self.model.wsol_synth[:,1], color=self.colors[1], **kwargs)
            if not label_set:
                line[0].set_label('ODE solution')
                label_set = True
        if 'R' in self.values_to_plot:
            line = ax.plot(self.model.t_synth, self.model.wsol_synth[:,2], color=self.colors[2], **kwargs)
            if not label_set:
                line[0].set_label('ODE solution')
                label_set = True
        if 'D' in self.values_to_plot:
            line = ax.plot(self.model.t_synth, self.model.wsol_synth[:,3], color=self.colors[3], **kwargs)
            if not label_set:
                line[0].set_label('ODE solution')
                label_set = True
        # self._set_color(line, self.colors)
    
    def _plot_pred_nn(self, ax, linestyle='--', **kwargs):
        label_set = False
        if 'S' in self.values_to_plot:
            line = ax.plot(self.model.t_nn_synth, self.model.wsol_nn_synth[:,0], linestyle=linestyle, color=self.colors[0], **kwargs)
            if not label_set:
                line[0].set_label('PINN prediction')
                label_set = True
        if 'I' in self.values_to_plot:
            line = ax.plot(self.model.t_nn_synth, self.model.wsol_nn_synth[:,1], linestyle=linestyle, color=self.colors[1], **kwargs)
            if not label_set:
                line[0].set_label('PINN prediction')
                label_set = True
        if 'R' in self.values_to_plot:
            line = ax.plot(self.model.t_nn_synth, self.model.wsol_nn_synth[:,2], linestyle=linestyle, color=self.colors[2], **kwargs)
            if not label_set:
                line[0].set_label('PINN prediction')
                label_set = True
        if 'D' in self.values_to_plot:
            line = ax.plot(self.model.t_nn_synth, self.model.wsol_nn_synth[:,3], linestyle=linestyle, color=self.colors[3], **kwargs)
            if not label_set:
                line[0].set_label('PINN prediction')
                label_set = True
        # self._set_color(line, self.colors)
    
    def plot_future_prediction(self, ax):
        ax.set_title('Future prediction')

        self._plot_pred_synthetic(ax)
        
        self._plot_pred_nn(ax)
        
        ax.legend()
    

class SIRD_deepxde_net:
    def __init__(self, t, wsol, alpha_guess=0.1, beta_guess=0.1, gamma_guess=0.1,
                 with_neumann=False, model_name=None, init_num_people=None):
        if model_name is None:
            model_name = 'variables'
        self.model_name = model_name
        
        self.t, self.wsol = t, wsol
        S_sol, I_sol, R_sol, D_sol = wsol[:,0], wsol[:,1], wsol[:,2], wsol[:,3]
        if init_num_people is None:
            init_num_people = np.sum(wsol[0,:])
        self.init_num_people = init_num_people
        S_sol, I_sol, R_sol, D_sol = S_sol/init_num_people, I_sol/init_num_people, R_sol/init_num_people, D_sol/init_num_people
        
        timedomain = dde.geometry.TimeDomain(0, max(t))
        
        alpha = dde.Variable(alpha_guess)
        beta = dde.Variable(beta_guess)
        gamma = dde.Variable(gamma_guess)
        
        def pde(t, y):
            S, I = y[:, 0:1], y[:, 1:2]
            
            dS_t = dde.grad.jacobian(y, t, i=0)
            dI_t = dde.grad.jacobian(y, t, i=1)
            dR_t = dde.grad.jacobian(y, t, i=2)
            dD_t = dde.grad.jacobian(y, t, i=3)
            
            return [dS_t + alpha*S*I, 
                   dI_t - alpha*S*I + beta*I + gamma*I,
                   dR_t - beta*I,
                   dD_t - gamma*I]
        
        def boundary(t_inp, on_initial):
            return on_initial and np.isclose(t_inp[0], t[0])
        
        def boundary_right(t_inp, on_final):
            # print(t[-1])
            return on_final and np.isclose(t_inp[0], t[-1])
        
        known_points = []
        
        # Initial conditions
        # TODO - BC for susceptible should be 1
        # TODO - BC for rest should be 0
        # ic_S = dde.icbc.IC(timedomain, lambda X: torch.tensor(S_sol[0]).reshape(1,1), boundary, component=0)
        # ic_I = dde.icbc.IC(timedomain, lambda X: torch.tensor(I_sol[0]).reshape(1,1), boundary, component=1)
        # ic_R = dde.icbc.IC(timedomain, lambda X: torch.tensor(R_sol[0]).reshape(1,1), boundary, component=2)
        # ic_D = dde.icbc.IC(timedomain, lambda X: torch.tensor(D_sol[0]).reshape(1,1), boundary, component=3)
        
        # known_points += [ic_S, ic_I, ic_R, ic_D]
        
        # Test points
        # TODO - how do we weight right points higher than earlier points?
        select_points_after_bool = t>=0
        t_later = t[select_points_after_bool]
        S_sol_later = S_sol[select_points_after_bool]
        I_sol_later = I_sol[select_points_after_bool]
        R_sol_later = R_sol[select_points_after_bool]
        D_sol_later = D_sol[select_points_after_bool]
        
        observe_S = dde.icbc.PointSetBC(t_later.reshape(len(t_later), 1), S_sol_later.reshape(len(S_sol_later), 1), component=0)
        observe_I = dde.icbc.PointSetBC(t_later.reshape(len(t_later), 1), I_sol_later.reshape(len(I_sol_later), 1), component=1)
        observe_R = dde.icbc.PointSetBC(t_later.reshape(len(t_later), 1), R_sol_later.reshape(len(R_sol_later), 1), component=2)
        observe_D = dde.icbc.PointSetBC(t_later.reshape(len(t_later), 1), D_sol_later.reshape(len(D_sol_later), 1), component=3)
        
        known_points += [observe_S, observe_I, observe_R,observe_D]
        
        # Final conditions
        # fc_S = dde.DirichletBC(timedomain, lambda X: torch.tensor(S_sol[-1]).reshape(1,1), boundary_right, component=0)
        # fc_I = dde.DirichletBC(timedomain, lambda X: torch.tensor(I_sol[-1]).reshape(1,1), boundary_right, component=1)
        # fc_R = dde.DirichletBC(timedomain, lambda X: torch.tensor(R_sol[-1]).reshape(1,1), boundary_right, component=2)
        # fc_D = dde.DirichletBC(timedomain, lambda X: torch.tensor(D_sol[-1]).reshape(1,1), boundary_right, component=3)
        
        # known_points += [fc_S, fc_I, fc_R, fc_D]
        
        # Neumann
        if with_neumann:
            S_diff = (S_sol[-1] - S_sol[-2]) / (t[-1] - t[-2])
            I_diff = (I_sol[-1] - I_sol[-2]) / (t[-1] - t[-2])
            R_diff = (R_sol[-1] - R_sol[-2]) / (t[-1] - t[-2])
            D_diff = (D_sol[-1] - D_sol[-2]) / (t[-1] - t[-2])
            
            fc_n_S = dde.NeumannBC(timedomain, lambda X: torch.tensor(S_diff).reshape(1,1), boundary_right, component=0)
            fc_n_I = dde.NeumannBC(timedomain, lambda X: torch.tensor(I_diff).reshape(1,1), boundary_right, component=1)
            fc_n_R = dde.NeumannBC(timedomain, lambda X: torch.tensor(R_diff).reshape(1,1), boundary_right, component=2)
            fc_n_D = dde.NeumannBC(timedomain, lambda X: torch.tensor(D_diff).reshape(1,1), boundary_right, component=3)
            
            known_points += [fc_n_S, fc_n_I, fc_n_R, fc_n_D]
        
        self.data = dde.data.PDE(
            timedomain,
            pde,
            known_points,
            num_domain=600, #TODO would it help to have this number higher? IE weight the result of the PDE higher
            num_boundary=10,
            anchors=t.reshape(len(t), 1),
        )
        self.variables = [alpha, beta, gamma]
    
    def set_synthetic_data(self, t, wsol):
        self.t_synth, self.wsol_synth = t, wsol
    
    def set_nn_synthetic_data(self, t, wsol):
        self.t_nn_synth, self.wsol_nn_synth = t, wsol
    
    def init_model(self, layer_size=None, activation="tanh", initializer="Glorot uniform", 
                   lr=0.001, optimizer="adam", print_every=100):
        
        if layer_size is None:
            layer_size = [1] + [32] * 3 + [4]
        
        net = dde.nn.FNN(layer_size, activation, initializer)
        
        self.model = dde.Model(self.data, net)
        
        #TODO - should we add decay here?
        #TODO - batch size (see https://github.com/lululxvi/deepxde/issues/320)
        self.model.compile(optimizer, lr=lr 
                           # ,metrics=["l2 relative error"]
                           ,loss="MSE"
                           ,external_trainable_variables=self.variables
                           #,loss_weights=[0.5,0.5,0.5,0.5,1,1,1,1]
                      )
        
        self.variable = dde.callbacks.VariableValue(self.variables, period=print_every, filename=f'{self.model_name}.txt')
    
    def train_model(self, iterations=7500, print_every=1000):
        self.losshistory, self.train_state = self.model.train(iterations=iterations, 
                                                              callbacks=[self.variable],
                                                              display_every=print_every)
        # self.alpha_nn, self.beta_nn, self.gamma_nn = self.variable.get_value()
        self._get_best_params()
        self._best_nn_prediction()
    
    def _get_best_params(self):
        df = pd.read_csv(f'{self.model_name}.txt', header=None, delimiter=' ', index_col=0)
        df[1] = df[1].str[1:-1].astype('float')
        df[2] = df[2].str[:-1].astype('float')
        df[3] = df[3].str[:-1].astype('float')
        df = df.loc[self.train_state.best_step]
        self.best_alpha_nn, self.best_beta_nn, self.best_gamma_nn = df[1], df[2], df[3]
        return self.best_alpha_nn, self.best_beta_nn, self.best_gamma_nn
    
    def get_best_params(self):
        return self.best_alpha_nn, self.best_beta_nn, self.best_gamma_nn
    
    # def get_predicted_params(self):
    #     return self.alpha_nn, self.beta_nn, self.gamma_nn
    
    def _best_nn_prediction(self):
        y_dim = self.train_state.best_y.shape[1]

        idx = np.argsort(self.train_state.X_test[:, 0])
        self.t_nn_best = self.train_state.X_test[idx, 0]
        
        wsol_nn_best = []
        for i in range(y_dim):
            wsol_nn_best.append(self.train_state.best_y[idx, i]*self.init_num_people)
        self.wsol_nn_best = np.array(wsol_nn_best).T
    
    def get_best_nn_prediction(self):
        return self.t_nn_best, self.wsol_nn_best
    
    def run_all(self, t_synth, wsol_synth, solver, print_every=1000, iterations=8000
                ,layer_size=None, activation="tanh", initializer="Glorot uniform"
                ,lr=0.001, optimizer="adam"):
        self.init_model(print_every=print_every, layer_size=layer_size, activation=activation,
                        initializer=initializer, lr=lr, optimizer=optimizer)
        self.train_model(iterations=iterations, print_every=print_every)
        alpha_nn, beta_nn, gamma_nn = self.get_best_params()
        t_nn_param, wsol_nn_param, N_nn_param = solver.solve_SIRD(alpha_nn, beta_nn, gamma_nn)
        self.set_synthetic_data(t_synth, wsol_synth)
        self.set_nn_synthetic_data(t_nn_param, wsol_nn_param)

if __name__=='__main__':
    seed = 0
    np.random.seed(seed)
    dde.config.set_random_seed(seed)
    
    alpha_real = 0.2
    beta_real = 0.05
    gamma_real = 0.01
    import ODE_SIR
    solver = ODE_SIR.ODESolver()
    t_synth, wsol_synth, N = solver.solve_SIRD(alpha_real, beta_real, gamma_real)
    t_bool = t_synth  < 85
    t, wsol = t_synth[t_bool], wsol_synth[t_bool]
    wsol = solver.add_noise(wsol, scale_pct=0.05)
    
    fig, ax = plt.subplots(dpi=300, figsize=(6,6))
    solver.plot_synthetic_and_sample(t_synth, wsol_synth, t, wsol, title='SIRD solution and train data', ax=ax)
    plt.savefig('SIRD solution',bbox_inches='tight')
    # model = SIRD_deepxde_net(t_synth, wsol_synth)
    
    # model.run_all(t_synth, wsol_synth, solver, iterations=10000)
    
    
    # values_to_plot = ['I']
    # plot_model = Plot(model, values_to_plot=values_to_plot)
    
    # fig, ax = plt.subplots()
    # line = ax.scatter(plot_model.model.t_nn_synth, plot_model.model.wsol_synth[:,1], color=plot_model.colors[0], label='True',alpha=0.5)
    # line = ax.plot(plot_model.model.t_synth, plot_model.model.wsol_nn_synth[:,1], color=plot_model.colors[1], label='2')
    
    # pred_x = np.arange(0,250)
    # pred_x = pred_x.reshape(len(pred_x), 1)
    # pred_y = model.model.predict(pred_x)
    
    # plt.plot(pred_x, pred_y)
    # plt.vlines(85, 0,1.2)


