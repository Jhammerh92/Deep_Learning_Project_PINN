import deepxde as dde
import numpy as np
import torch
import matplotlib.pyplot as plt

class Plot:
    def __init__(self, model, colors = None):
        self.model = model
        
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
        colors = self.colors
        line = ax.plot(self.model.t, self.model.wsol)
        line[0].set_label('Synthetic')
        self._set_color(line, self.colors)
    
    def _plot_known_nn(self, ax, linestyle='--'):
        line = ax.plot(self.model.t_nn_best, self.model.wsol_nn_best, linestyle=linestyle)
        line[0].set_label('PINN prediction')
        self._set_color(line, self.colors)
    
    def plot_known_data(self, ax):
        ax.set_title('Known data')
        
        self._plot_known_synthetic(ax)
        
        self._plot_known_nn(ax)
        
        ax.legend()
    
    def _plot_pred_synthetic(self, ax):
        line = ax.plot(self.model.t_synth, self.model.wsol_synth)
        line[0].set_label('Synthetic')
        self._set_color(line, self.colors)
    
    def _plot_pred_nn(self, ax, linestyle='--'):
        line = ax.plot(self.model.t_nn_synth, self.model.wsol_nn_synth, linestyle=linestyle)
        line[0].set_label('Prediction')
        self._set_color(line, self.colors)
    
    def plot_future_prediction(self, ax):
        ax.set_title('Future prediction')

        self._plot_pred_synthetic(ax)
        
        self._plot_pred_nn(ax)
        
        ax.legend()
    

class SIRD_deepxde_net:
    def __init__(self, t, wsol, alpha_guess=0.1, beta_guess=0.1, gamma_guess=0.1):
        self.t, self.wsol = t, wsol
        S_sol, I_sol, R_sol, D_sol = wsol[:,0], wsol[:,1], wsol[:,2], wsol[:,3]
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
        
        # Initial conditions
        ic_S = dde.icbc.IC(timedomain, lambda X: torch.tensor(S_sol[0]).reshape(1,1), boundary, component=0)
        ic_I = dde.icbc.IC(timedomain, lambda X: torch.tensor(I_sol[0]).reshape(1,1), boundary, component=1)
        ic_R = dde.icbc.IC(timedomain, lambda X: torch.tensor(R_sol[0]).reshape(1,1), boundary, component=2)
        ic_D = dde.icbc.IC(timedomain, lambda X: torch.tensor(D_sol[0]).reshape(1,1), boundary, component=3)
        
        # Test points
        # TODO - how do we weight right points higher than earlier points?
        observe_S = dde.icbc.PointSetBC(t.reshape(len(t), 1), S_sol.reshape(len(S_sol), 1), component=0)
        observe_I = dde.icbc.PointSetBC(t.reshape(len(t), 1), I_sol.reshape(len(I_sol), 1), component=1)
        observe_R = dde.icbc.PointSetBC(t.reshape(len(t), 1), R_sol.reshape(len(R_sol), 1), component=2)
        observe_D = dde.icbc.PointSetBC(t.reshape(len(t), 1), D_sol.reshape(len(D_sol), 1), component=3)
        
        # Final conditions
        # TODO - we want to use cauchy BC (I think :) )
        # find derivative using finite differences
        fc_S = dde.DirichletBC(timedomain, lambda X: torch.tensor(S_sol[-1]).reshape(1,1), boundary_right, component=0)
        fc_I = dde.DirichletBC(timedomain, lambda X: torch.tensor(I_sol[-1]).reshape(1,1), boundary_right, component=1)
        fc_R = dde.DirichletBC(timedomain, lambda X: torch.tensor(R_sol[-1]).reshape(1,1), boundary_right, component=2)
        fc_D = dde.DirichletBC(timedomain, lambda X: torch.tensor(D_sol[-1]).reshape(1,1), boundary_right, component=3)
        
        S_diff = (S_sol[-1] - S_sol[-2]) / (t[-1] - t[-2])
        I_diff = (I_sol[-1] - I_sol[-2]) / (t[-1] - t[-2])
        R_diff = (R_sol[-1] - R_sol[-2]) / (t[-1] - t[-2])
        D_diff = (D_sol[-1] - D_sol[-2]) / (t[-1] - t[-2])
        
        fc_n_S = dde.NeumannBC(timedomain, lambda X: torch.tensor(S_diff).reshape(1,1), boundary_right, component=0)
        fc_n_I = dde.NeumannBC(timedomain, lambda X: torch.tensor(I_diff).reshape(1,1), boundary_right, component=1)
        fc_n_R = dde.NeumannBC(timedomain, lambda X: torch.tensor(R_diff).reshape(1,1), boundary_right, component=2)
        fc_n_D = dde.NeumannBC(timedomain, lambda X: torch.tensor(D_diff).reshape(1,1), boundary_right, component=3)
        
        self.data = dde.data.PDE(
            timedomain,
            pde,
            [ic_S, ic_I, ic_R, ic_D, 
             observe_S, observe_I,
             observe_R,observe_D
            ,fc_S, fc_I, fc_R, fc_D
            ,fc_n_S, fc_n_I, fc_n_R, fc_n_D
             ],
            num_domain=50,
            num_boundary=10,
            anchors=t.reshape(len(t), 1),
        )
        self.variables = [alpha, beta, gamma]
    
    def set_synthetic_data(self, t, wsol):
        self.t_synth, self.wsol_synth = t, wsol
    
    def set_nn_synthetic_data(self, t, wsol):
        self.t_nn_synth, self.wsol_nn_synth = t, wsol
    
    def init_model(self, layer_size=None, activation="tanh", initializer="Glorot uniform", 
                   lr=0.01, optimizer="adam", print_every=100):
        if layer_size is None:
            layer_size = [1] + [32] * 4 + [4]
        
        net = dde.nn.FNN(layer_size, activation, initializer)
        
        self.model = dde.Model(self.data, net)
        
        self.model.compile(optimizer, lr=lr 
                           # ,metrics=["l2 relative error"]
                           ,loss="MSE"
                           ,external_trainable_variables=self.variables
                           #,loss_weights=[0.5,0.5,0.5,0.5,1,1,1,1]
                      )
        
        self.variable = dde.callbacks.VariableValue(self.variables, period=print_every)
    
    def train_model(self, iterations=7500):
        self.losshistory, self.train_state = self.model.train(iterations=iterations, callbacks=[self.variable])
        self.alpha_nn, self.beta_nn, self.gamma_nn = self.variable.get_value()
        self._best_nn_prediction()
        
    def get_predicted_params(self):
        return self.alpha_nn, self.beta_nn, self.gamma_nn
    
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