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
        s = 1
        if 'S' in self.values_to_plot:
            line = ax.scatter(self.model.t, self.model.wsol[:,0], color=self.colors[0], s=s)
        if 'I' in self.values_to_plot:
            line = ax.scatter(self.model.t, self.model.wsol[:,1], color=self.colors[1], s=s)
        if 'R' in self.values_to_plot:
            line = ax.scatter(self.model.t, self.model.wsol[:,2], color=self.colors[2], s=s)
        if 'D' in self.values_to_plot:
            line = ax.scatter(self.model.t, self.model.wsol[:,3], color=self.colors[3], s=s)
        line.set_label('Synthetic')
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
    
    def _plot_pred_synthetic(self, ax):
        if 'S' in self.values_to_plot:
            line = ax.plot(self.model.t_synth, self.model.wsol_synth[:,0], color=self.colors[0])
        if 'I' in self.values_to_plot:
            line = ax.plot(self.model.t_synth, self.model.wsol_synth[:,1], color=self.colors[1])
        if 'R' in self.values_to_plot:
            line = ax.plot(self.model.t_synth, self.model.wsol_synth[:,2], color=self.colors[2])
        if 'D' in self.values_to_plot:
            line = ax.plot(self.model.t_synth, self.model.wsol_synth[:,3], color=self.colors[3])
        
        line[0].set_label('Synthetic')
        # self._set_color(line, self.colors)
    
    def _plot_pred_nn(self, ax, linestyle='--'):
        if 'S' in self.values_to_plot:
            line = ax.plot(self.model.t_nn_synth, self.model.wsol_nn_synth[:,0], linestyle=linestyle, color=self.colors[0])
        if 'I' in self.values_to_plot:
            line = ax.plot(self.model.t_nn_synth, self.model.wsol_nn_synth[:,1], linestyle=linestyle, color=self.colors[1])
        if 'R' in self.values_to_plot:
            line = ax.plot(self.model.t_nn_synth, self.model.wsol_nn_synth[:,2], linestyle=linestyle, color=self.colors[2])
        if 'D' in self.values_to_plot:
            line = ax.plot(self.model.t_nn_synth, self.model.wsol_nn_synth[:,3], linestyle=linestyle, color=self.colors[3])
        line[0].set_label('Prediction')
        # self._set_color(line, self.colors)
    
    def plot_future_prediction(self, ax):
        ax.set_title('Future prediction')

        self._plot_pred_synthetic(ax)
        
        self._plot_pred_nn(ax)
        
        ax.legend()
    

class SIRD_deepxde_net:
    def __init__(self, t, wsol, alpha_guess=0.1, beta_guess=0.1, gamma_guess=0.1,
                 with_neumann=False):
        self.t, self.wsol = t, wsol
        S_sol, I_a_sol, I_b_sol, I_aa_sol, I_bb_sol, I_ba_sol, I_ab_sol, R_a_sol, R_b_sol, R_sol, D_sol = wsol[:,0], wsol[:,1], wsol[:,2], wsol[:,3], wsol[:,4], wsol[:,5], wsol[:,6], wsol[:,7], wsol[:,8], wsol[:,9], wsol[:,3], wsol[:,3]
        init_num_people = np.sum(wsol[0,:])
        self.init_num_people = init_num_people
        S_sol, I_a_sol, I_b_sol, I_aa_sol, I_bb_sol, I_ba_sol, I_ab_sol, R_a_sol, R_b_sol, R_sol, D_sol = S_sol/init_num_people, I_a_sol/init_num_people, I_b_sol/init_num_people, I_aa_sol/init_num_people, I_bb_sol/init_num_people, I_ba_sol/init_num_people, I_ab_sol/init_num_people, R_a_sol/init_num_people, R_b_sol/init_num_people, R_sol/init_num_people, D_sol/init_num_people
        
        timedomain = dde.geometry.TimeDomain(0, max(t))
        
        alpha_a = dde.Variable(alpha_guess)
        alpha_aa = alpha_a*0.5
        alpha_ba = alpha_a*0.9
        beta_a = dde.Variable(beta_guess)
        gamma_a = dde.Variable(gamma_guess)

        alpha_b = dde.Variable(alpha_guess)
        alpha_bb = alpha_b*0.5
        alpha_ab = alpha_b*0.9
        beta_b = dde.Variable(beta_guess)
        gamma_b = dde.Variable(gamma_guess)
        
        def pde(t, y):
            # S, I = y[:, 0:1], y[:, 1:2]
            S, I_a, I_b, I_aa, I_bb, I_ba, I_ab, R_a, R_b = y[:, 0:9]
            
            dS_t = dde.grad.jacobian(y, t, i=0)
            dI_a_t = dde.grad.jacobian(y, t, i=1)
            dI_b_t = dde.grad.jacobian(y, t, i=1)
            dI_aa_t = dde.grad.jacobian(y, t, i=1)
            dI_bb_t = dde.grad.jacobian(y, t, i=1)
            dI_ba_t = dde.grad.jacobian(y, t, i=1)
            dI_ab_t = dde.grad.jacobian(y, t, i=1)
            dR_a_t = dde.grad.jacobian(y, t, i=2)
            dR_b_t = dde.grad.jacobian(y, t, i=2)
            dR_t = dde.grad.jacobian(y, t, i=2)
            dD_t = dde.grad.jacobian(y, t, i=3)
            
            return [dS_t + alpha_a*S*I_a + alpha_b*S*I_b, 
                   dI_a_t - alpha_a*S*I_a + beta_a*I_a + gamma_a*I_a,
                   dI_b_t - alpha_b*S*I_b + beta_b*I_b + gamma_b*I_b,
                   dI_aa_t - alpha_aa*R_a*(I_a + I_aa + I_ba) + beta_a*I_aa + gamma_a*I_aa,
                   dI_bb_t - alpha_bb*R_b*(I_b + I_bb + I_ab) + beta_b*I_bb + gamma_b*I_bb,
                   dI_ba_t - alpha_ba*R_b*(I_a + I_aa + I_ba) + beta_a*I_ba + gamma_a*I_ba,
                   dI_ab_t - alpha_ab*R_a*(I_b + I_bb + I_ab) + beta_b*I_ab + gamma_b*I_ab,
                   dR_a_t - beta_a*I_a + alpha_aa*R_a*(I_a + I_aa + I_ba) + alpha_ab*R_a*(I_b + I_bb + I_ab),
                   dR_b_t - beta_b*I_b + alpha_bb*R_b*(I_b + I_bb + I_ab) + alpha_ba*R_b*(I_a + I_aa + I_ba),
                   dR_t - beta_a*(I_aa + I_ba) - beta_b*(I_bb + I_ab),
                   dD_t - gamma_a*(I_a + I_aa + I_ba) - gamma_b*(I_b + I_bb + I_ab)]
        
        def boundary(t_inp, on_initial):
            return on_initial and np.isclose(t_inp[0], t[0])
        
        def boundary_right(t_inp, on_final):
            # print(t[-1])
            return on_final and np.isclose(t_inp[0], t[-1])
        
        known_points = []
        
        # Initial conditions
        ic_S = dde.icbc.IC(timedomain, lambda X: torch.tensor(S_sol[0]).reshape(1,1), boundary, component=0)
        ic_I_a = dde.icbc.IC(timedomain, lambda X: torch.tensor(I_a_sol[0]).reshape(1,1), boundary, component=1)
        ic_I_b = dde.icbc.IC(timedomain, lambda X: torch.tensor(I_b_sol[0]).reshape(1,1), boundary, component=1)
        ic_I_aa = dde.icbc.IC(timedomain, lambda X: torch.tensor(I_aa_sol[0]).reshape(1,1), boundary, component=1)
        ic_I_bb = dde.icbc.IC(timedomain, lambda X: torch.tensor(I_bb_sol[0]).reshape(1,1), boundary, component=1)
        ic_I_ba = dde.icbc.IC(timedomain, lambda X: torch.tensor(I_ba_sol[0]).reshape(1,1), boundary, component=1)
        ic_I_ab = dde.icbc.IC(timedomain, lambda X: torch.tensor(I_ab_sol[0]).reshape(1,1), boundary, component=1)
        ic_R_a = dde.icbc.IC(timedomain, lambda X: torch.tensor(R_a_sol[0]).reshape(1,1), boundary, component=2)
        ic_R_b = dde.icbc.IC(timedomain, lambda X: torch.tensor(R_b_sol[0]).reshape(1,1), boundary, component=2)
        ic_R = dde.icbc.IC(timedomain, lambda X: torch.tensor(R_sol[0]).reshape(1,1), boundary, component=2)
        ic_D = dde.icbc.IC(timedomain, lambda X: torch.tensor(D_sol[0]).reshape(1,1), boundary, component=3)
        
        known_points += [ic_S, ic_I_a, ic_I_b, ic_I_aa, ic_I_bb, ic_I_ba, ic_I_ab, ic_R_a, ic_R_b, ic_R, ic_D]
        
        # Test points
        # TODO - how do we weight right points higher than earlier points?
        select_points_after_bool = t>=0
        t_later = t[select_points_after_bool]
        S_sol_later = S_sol[select_points_after_bool]
        I_a_sol_later = I_a_sol[select_points_after_bool]
        I_b_sol_later = I_b_sol[select_points_after_bool]
        I_aa_sol_later = I_aa_sol[select_points_after_bool]
        I_bb_sol_later = I_bb_sol[select_points_after_bool]
        I_ba_sol_later = I_ba_sol[select_points_after_bool]
        I_ab_sol_later = I_ab_sol[select_points_after_bool]
        R_a_sol_later = R_a_sol[select_points_after_bool]
        R_b_sol_later = R_b_sol[select_points_after_bool]
        R_sol_later = R_sol[select_points_after_bool]
        D_sol_later = D_sol[select_points_after_bool]
        
        observe_S = dde.icbc.PointSetBC(t_later.reshape(len(t_later), 1), S_sol_later.reshape(len(S_sol_later), 1), component=0)
        observe_I_a = dde.icbc.PointSetBC(t_later.reshape(len(t_later), 1), I_a_sol_later.reshape(len(I_a_sol_later), 1), component=1)
        observe_I_b = dde.icbc.PointSetBC(t_later.reshape(len(t_later), 1), I_b_sol_later.reshape(len(I_b_sol_later), 1), component=1)
        observe_I_aa = dde.icbc.PointSetBC(t_later.reshape(len(t_later), 1), I_aa_sol_later.reshape(len(I_aa_sol_later), 1), component=1)
        observe_I_bb = dde.icbc.PointSetBC(t_later.reshape(len(t_later), 1), I_bb_sol_later.reshape(len(I_bb_sol_later), 1), component=1)
        observe_I_ba = dde.icbc.PointSetBC(t_later.reshape(len(t_later), 1), I_ba_sol_later.reshape(len(I_ba_sol_later), 1), component=1)
        observe_I_ab = dde.icbc.PointSetBC(t_later.reshape(len(t_later), 1), I_ab_sol_later.reshape(len(I_ab_sol_later), 1), component=1)
        observe_R_a = dde.icbc.PointSetBC(t_later.reshape(len(t_later), 1), R_a_sol_later.reshape(len(R_a_sol_later), 1), component=2)
        observe_R_b = dde.icbc.PointSetBC(t_later.reshape(len(t_later), 1), R_b_sol_later.reshape(len(R_b_sol_later), 1), component=2)
        observe_R = dde.icbc.PointSetBC(t_later.reshape(len(t_later), 1), R_sol_later.reshape(len(R_sol_later), 1), component=2)
        observe_D = dde.icbc.PointSetBC(t_later.reshape(len(t_later), 1), D_sol_later.reshape(len(D_sol_later), 1), component=3)
        
        known_points += [observe_S,
                         observe_I_a,
                         observe_I_b,
                         observe_I_aa,
                         observe_I_bb,
                         observe_I_ba,
                         observe_I_ab,
                         observe_R_a,
                         observe_R_b,
                         observe_R,
                         observe_D]
        
        # Final conditions
        fc_S = dde.DirichletBC(timedomain, lambda X: torch.tensor(S_sol[-1]).reshape(1,1), boundary_right, component=0)
        fc_I_a = dde.DirichletBC(timedomain, lambda X: torch.tensor(I_a_sol[-1]).reshape(1,1), boundary_right, component=1)
        fc_I_b = dde.DirichletBC(timedomain, lambda X: torch.tensor(I_b_sol[-1]).reshape(1,1), boundary_right, component=1)
        fc_I_aa = dde.DirichletBC(timedomain, lambda X: torch.tensor(I_aa_sol[-1]).reshape(1,1), boundary_right, component=1)
        fc_I_bb = dde.DirichletBC(timedomain, lambda X: torch.tensor(I_bb_sol[-1]).reshape(1,1), boundary_right, component=1)
        fc_I_ba = dde.DirichletBC(timedomain, lambda X: torch.tensor(I_ba_sol[-1]).reshape(1,1), boundary_right, component=1)
        fc_I_ab = dde.DirichletBC(timedomain, lambda X: torch.tensor(I_ab_sol[-1]).reshape(1,1), boundary_right, component=1)
        fc_R_a = dde.DirichletBC(timedomain, lambda X: torch.tensor(R_a_sol[-1]).reshape(1,1), boundary_right, component=2)
        fc_R_b = dde.DirichletBC(timedomain, lambda X: torch.tensor(R_b_sol[-1]).reshape(1,1), boundary_right, component=2)
        fc_R = dde.DirichletBC(timedomain, lambda X: torch.tensor(R_sol[-1]).reshape(1,1), boundary_right, component=2)
        fc_D = dde.DirichletBC(timedomain, lambda X: torch.tensor(D_sol[-1]).reshape(1,1), boundary_right, component=3)
        
        known_points += [fc_S,
                         fc_I_a, 
                         fc_I_b, 
                         fc_I_aa, 
                         fc_I_bb, 
                         fc_I_ba, 
                         fc_I_ab, 
                         fc_R_a, 
                         fc_R_b, 
                         fc_R, 
                         fc_D]
        
        # Neumann
        if with_neumann:
            S_diff = (S_sol[-1] - S_sol[-2]) / (t[-1] - t[-2])
            I_a_diff = (I_a_sol[-1] - I_a_sol[-2]) / (t[-1] - t[-2])
            I_b_diff = (I_b_sol[-1] - I_b_sol[-2]) / (t[-1] - t[-2])
            I_aa_diff = (I_aa_sol[-1] - I_aa_sol[-2]) / (t[-1] - t[-2])
            I_bb_diff = (I_bb_sol[-1] - I_bb_sol[-2]) / (t[-1] - t[-2])
            I_ba_diff = (I_ba_sol[-1] - I_ba_sol[-2]) / (t[-1] - t[-2])
            I_ab_diff = (I_ab_sol[-1] - I_ab_sol[-2]) / (t[-1] - t[-2])
            R_a_diff = (R_a_sol[-1] - R_a_sol[-2]) / (t[-1] - t[-2])
            R_b_diff = (R_b_sol[-1] - R_b_sol[-2]) / (t[-1] - t[-2])
            R_diff = (R_sol[-1] - R_sol[-2]) / (t[-1] - t[-2])
            D_diff = (D_sol[-1] - D_sol[-2]) / (t[-1] - t[-2])
            
            fc_n_S = dde.NeumannBC(timedomain, lambda X: torch.tensor(S_diff).reshape(1,1), boundary_right, component=0)
            fc_n_Ia_ = dde.NeumannBC(timedomain, lambda X: torch.tensor(I_a_diff).reshape(1,1), boundary_right, component=1)
            fc_n_Ib_ = dde.NeumannBC(timedomain, lambda X: torch.tensor(I_b_diff).reshape(1,1), boundary_right, component=1)
            fc_n_Iaa = dde.NeumannBC(timedomain, lambda X: torch.tensor(I_aa_diff).reshape(1,1), boundary_right, component=1)
            fc_n_Ibb = dde.NeumannBC(timedomain, lambda X: torch.tensor(I_bb_diff).reshape(1,1), boundary_right, component=1)
            fc_n_Iba = dde.NeumannBC(timedomain, lambda X: torch.tensor(I_ba_diff).reshape(1,1), boundary_right, component=1)
            fc_n_Iab = dde.NeumannBC(timedomain, lambda X: torch.tensor(I_ab_diff).reshape(1,1), boundary_right, component=1)
            fc_n_Ra_ = dde.NeumannBC(timedomain, lambda X: torch.tensor(R_a_diff).reshape(1,1), boundary_right, component=2)
            fc_n_Rb_ = dde.NeumannBC(timedomain, lambda X: torch.tensor(R_b_diff).reshape(1,1), boundary_right, component=2)
            fc_n_R = dde.NeumannBC(timedomain, lambda X: torch.tensor(R_diff).reshape(1,1), boundary_right, component=2)
            fc_n_D = dde.NeumannBC(timedomain, lambda X: torch.tensor(D_diff).reshape(1,1), boundary_right, component=3)
            
            known_points += [fc_S,
                            fc_I_a, 
                            fc_I_b, 
                            fc_I_aa, 
                            fc_I_bb, 
                            fc_I_ba, 
                            fc_I_ab, 
                            fc_R_a, 
                            fc_R_b, 
                            fc_R, 
                            fc_D]
            # known_points += [fc_n_S, fc_n_I, fc_n_R, fc_n_D]
        
        self.data = dde.data.PDE(
            timedomain,
            pde,
            known_points,
            num_domain=50,
            num_boundary=10,
            anchors=t.reshape(len(t), 1),
        )
        self.variables = [alpha_a, beta_a, gamma_a, alpha_b, beta_b, gamma_b]
    
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
        
        #TODO - should we add decay here?
        #TODO - batch size (see https://github.com/lululxvi/deepxde/issues/320)
        self.model.compile(optimizer, lr=lr 
                           # ,metrics=["l2 relative error"]
                           ,loss="MSE"
                           ,external_trainable_variables=self.variables
                           #,loss_weights=[0.5,0.5,0.5,0.5,1,1,1,1]
                      )
        
        self.variable = dde.callbacks.VariableValue(self.variables, period=print_every, filename='variables.txt')
    
    def train_model(self, iterations=7500, print_every=1000):
        self.losshistory, self.train_state = self.model.train(iterations=iterations, 
                                                              callbacks=[self.variable],
                                                              display_every=print_every)
        # self.alpha_nn, self.beta_nn, self.gamma_nn = self.variable.get_value()
        self._get_best_params()
        self._best_nn_prediction()
    
    def _get_best_params(self):
        df = pd.read_csv('variables.txt', header=None, delimiter=' ', index_col=0)
        # lav et loop af en art?
        for i, d in enumerate(df):
            df[i] = d.str[1:-1].astype('float')

        # df[1] = df[1].str[1:-1].astype('float')
        # df[2] = df[2].str[:-1].astype('float')
        # df[3] = df[3].str[:-1].astype('float')
        df = df.loc[self.train_state.best_step]
        # self.best_alpha_nn, self.best_beta_nn, self.best_gamma_nn = df[1], df[2], df[3]
        self.best_alpha_a_nn, self.best_alpha_b_nn, self.best_beta_a_nn, self.best_beta_b_nn, self.best_gamma_a_nn, self.best_gamma_b_nn = tuple(df)
        return self.best_alpha_a_nn, self.best_alpha_b_nn, self.best_beta_a_nn, self.best_beta_b_nn, self.best_gamma_a_nn, self.best_gamma_b_nn
    
    def get_best_params(self):
        return self.best_alpha_a_nn, self.best_alpha_b_nn, self.best_beta_a_nn, self.best_beta_b_nn, self.best_gamma_a_nn, self.best_gamma_b_nn
    
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
                ,lr=0.01, optimizer="adam"):
        self.init_model(print_every=print_every, layer_size=layer_size, activation=activation,
                        initializer=initializer, lr=lr, optimizer=optimizer)
        self.train_model(iterations=iterations, print_every=print_every)
        alpha_nn, beta_nn, gamma_nn = self.get_best_params()
        t_nn_param, wsol_nn_param, N_nn_param = solver.solve_SIRD(alpha_nn, beta_nn, gamma_nn)
        self.set_synthetic_data(t_synth, wsol_synth)
        self.set_nn_synthetic_data(t_nn_param, wsol_nn_param)

if __name__=='__main__':
    """  does not work now i guess
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
    
    """
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


