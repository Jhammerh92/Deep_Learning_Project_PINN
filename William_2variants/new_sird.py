# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 19:44:11 2022

@author: willi
"""

from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
import deepxde as dde
import pandas as pd

class SIRD:
    def __init__(self):
       pass

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
        alpha_a, alpha_b, beta_a, beta_b, gamma = p
       
        f = [ - (alpha_a/(self.N) * I_a + alpha_b/(self.N) * I_b ) * S,
            (alpha_a/(self.N)) * S * I_a - beta_a * I_a - gamma * I_a, # should use all infected?? (I_a + I_aa + I_ba)
            (alpha_b/(self.N)) * S * I_b - beta_b * I_b - gamma * I_b,
            beta_a * I_a + beta_b * I_b,
            gamma * I_a + gamma * I_b,
            ]
        return f

    def solve_SIRD(self, s):
        # Initial conditions
        S, I_a, I_b, R, D = s['S'], s['I_a'], s['I_b'],  s['R'], s['D']
        
        alpha_a, alpha_b = s['alpha_a'], s['alpha_b']
        beta_a, beta_b = s['beta_a'], s['beta_b']
        gamma = s['gamma']
        
        self.N = S + I_a + I_b + R + D
        
        # ODE solver parameters
        abserr = 1.0e-8
        relerr = 1.0e-6
        stoptime = 600.0
        # numpoints = stoptime * 10
        numpoints = 600
        
        # Create the time samples for the output of the ODE solver.
        # I use a large number of points, only because I want to make
        # a plot of the solution that looks nice.
        # t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]
        t = np.linspace(0, stoptime, numpoints, endpoint=True)
        # Pack up the parameters and initial conditions:
        p = [alpha_a, alpha_b, beta_a, beta_b, gamma]
        w0 = [S, I_a, I_b, R, D]
        
        # Call the ODE solver.
        wsol = odeint(self.vectorfield, w0, t, args=(p,),
                      atol=abserr, rtol=relerr)
        return t, wsol
    
    def plot_SIRD(self, t, wsol, ax=None, title=None):
        # print("total ",(wsol[-1,:]) )
        # print("total ",np.sum(wsol[-1,:]) )
        if ax is None:
            fig, ax = plt.subplots()
        
        S = wsol[:,0]
        I_a = wsol[:,1]
        I_b = wsol[:,2]
        I = I_a + I_b
        R = wsol[:,3]
        D = wsol[:,4]
        
        ax.plot(t, S, label='S')
        ax.plot(t, I_a, label='I_a')
        ax.plot(t, I_b, label='I_b')
        ax.plot(t, I, alpha=0.5, linestyle='--', label='I')        
        ax.plot(t, R, label='R')
        ax.plot(t, D, label='D')
        
        ax.legend()
        ax.grid()
        # # self._axis_SIRD(ax)
        if title is not None:
            ax.set_title(title)

class SIRD_deepxde_net:
    def __init__(self, t, wsol, alpha_a=0.1, alpha_b=0.1,
                 beta_a=0.01, beta_b=0.01,
                 gamma=0.001):
        self.t, self.wsol = t, wsol
        S_sol, I_sol, R_sol, D_sol = wsol[:,0], wsol[:,1], wsol[:,2], wsol[:,3]
        init_num_people = np.sum(wsol[0,:])
        self.init_num_people = init_num_people
        S_sol, I_sol, R_sol, D_sol = S_sol/init_num_people, I_sol/init_num_people, R_sol/init_num_people, D_sol/init_num_people
        
        timedomain = dde.geometry.TimeDomain(0, max(t))
        
        alpha_a = dde.Variable(alpha_a)
        alpha_b = dde.Variable(alpha_b)
        
        beta_a = dde.Variable(beta_a)
        beta_b = dde.Variable(beta_b)
        
        gamma = dde.Variable(gamma)
        
        def pde(t, y):
            S, I_a, I_b = y[:, 0:1], y[:, 1:2], y[:, 2:3]
            
            dS_t = dde.grad.jacobian(y, t, i=0)
            dIa_t = dde.grad.jacobian(y, t, i=1)
            dIb_t = dde.grad.jacobian(y, t, i=2)
            dR_t = dde.grad.jacobian(y, t, i=3)
            dD_t = dde.grad.jacobian(y, t, i=4)
            
            expanded_sird = [dS_t + (alpha_a*I_a + alpha_b*I_b)*S,
                             dIa_t - alpha_a*S*I_a + beta_a*I_a + gamma*I_a,
                             dIb_t - alpha_b*S*I_b + beta_b*I_b + gamma*I_b,
                             dR_t - beta_a*I_a - beta_b*I_b,
                             dD_t - gamma*I_a - gamma*I_b
                             ]
            
            sird = [expanded_sird[0],
                    # sum(expanded_sird[1:3]),
                    expanded_sird[1],
                    expanded_sird[2],
                    expanded_sird[3],
                    expanded_sird[4]]
            
            return sird
            
        def boundary(t_inp, on_initial):
            return on_initial and np.isclose(t_inp[0], t[0])
        
        def boundary_right(t_inp, on_final):
            # print(t[-1])
            return on_final and np.isclose(t_inp[0], t[-1])
        
        known_points = []
        
        # Initial conditions
        # ic_S = dde.icbc.IC(timedomain, lambda X: torch.tensor(S_sol[0]).reshape(1,1), boundary, component=0)
        # ic_I = dde.icbc.IC(timedomain, lambda X: torch.tensor(I_sol[0]).reshape(1,1), boundary, component=1)
        # ic_R = dde.icbc.IC(timedomain, lambda X: torch.tensor(R_sol[0]).reshape(1,1), boundary, component=2)
        # ic_D = dde.icbc.IC(timedomain, lambda X: torch.tensor(D_sol[0]).reshape(1,1), boundary, component=3)
        
        # known_points += [ic_S, ic_I, ic_R, ic_D]
        
        # Test points
        observe_S = dde.icbc.PointSetBC(t.reshape(len(t), 1), S_sol.reshape(len(S_sol), 1), component=0)
        # observe_Ia = dde.icbc.PointSetBC(t.reshape(len(t), 1), I_sol.reshape(len(I_sol), 1), component=1)
        observe_I = dde.icbc.PointSetBC(t.reshape(len(t), 1), I_sol.reshape(len(I_sol), 1), component=[1,2])
        observe_R = dde.icbc.PointSetBC(t.reshape(len(t), 1), R_sol.reshape(len(R_sol), 1), component=3)
        observe_D = dde.icbc.PointSetBC(t.reshape(len(t), 1), D_sol.reshape(len(D_sol), 1), component=4)
        
        known_points += [observe_S,
                         observe_I, 
                         observe_R,
                         observe_D]
        
        # Final conditions
        # fc_S = dde.DirichletBC(timedomain, lambda X: torch.tensor(S_sol[-1]).reshape(1,1), boundary_right, component=0)
        # fc_I = dde.DirichletBC(timedomain, lambda X: torch.tensor(I_sol[-1]).reshape(1,1), boundary_right, component=1)
        # fc_R = dde.DirichletBC(timedomain, lambda X: torch.tensor(R_sol[-1]).reshape(1,1), boundary_right, component=2)
        # fc_D = dde.DirichletBC(timedomain, lambda X: torch.tensor(D_sol[-1]).reshape(1,1), boundary_right, component=3)
        
        # known_points += [fc_S, fc_I, fc_R, fc_D]
        
        self.data = dde.data.PDE(
            timedomain,
            pde,
            known_points,
            num_domain=600,
            num_boundary=10,
            anchors=t.reshape(len(t), 1),
        )
        self.variables = [alpha_a, alpha_b, beta_a, beta_b, gamma]
    
    def set_synthetic_data(self, t, wsol):
        self.t_synth, self.wsol_synth = t, wsol
    
    def set_nn_synthetic_data(self, t, wsol):
        self.t_nn_synth, self.wsol_nn_synth = t, wsol
    
    def init_model(self, layer_size=None, activation="tanh", initializer="Glorot uniform", 
                   lr=0.01, optimizer="adam", print_every=100):
        if layer_size is None:
            layer_size = [1] + [32] * 3 + [5]
        
        net = dde.nn.FNN(layer_size, activation, initializer)
        
        self.model = dde.Model(self.data, net)
        
        #TODO - should we add decay here?
        #TODO - batch size (see https://github.com/lululxvi/deepxde/issues/320)
        self.model.compile(optimizer, lr=lr 
                           # ,metrics=["l2 relative error"]
                           ,loss="MSE"
                           ,external_trainable_variables=self.variables
                            # ,loss_weights=[1,1,1,1,
                            #                1,1,1,1]
                            ,with_softadapt=True
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
        df[1] = df[1].str[1:-1].astype('float')
        df[2] = df[2].str[:-1].astype('float')
        df[3] = df[3].str[:-1].astype('float')
        df[4] = df[4].str[:-1].astype('float')
        df[5] = df[5].str[:-1].astype('float')
        # df[6] = df[6].str[:-1].astype('float')
        df = df.loc[self.train_state.best_step]
        self.best_params = [df[1], df[2], df[3], df[4], df[5]] #, df[6]
        return df[1], df[2], df[3], df[4], df[5]#, df[6]
    
    def get_best_params(self):
        return self.best_params
    
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
    seed = 0
    np.random.seed(seed)
    dde.config.set_random_seed(seed)
    
    s = {'S':5000000,
         'I_a':50,
         'I_b':50,
         'R':0,
         'D':0,
         
         'alpha_a':0.05,
         'beta_a':0.0075,
         
         'alpha_b':0.1,
         'beta_b':0.05,
         
         'gamma':0.0005
         }
    
    solver = SIRD()
    t, wsol = solver.solve_SIRD(s)
    
    S = wsol[:,0]
    I = np.sum(wsol[:,1:3], axis=1)
    R = wsol[:,3]
    D = wsol[:,4]
    sird = np.c_[S,I,R,D]
    
    # model = SIRD_deepxde_net(t, wsol)
    model = SIRD_deepxde_net(t, sird,
                             alpha_a=s['alpha_a'], beta_a=s['beta_a'],
                             alpha_b=s['alpha_b'], beta_b=s['beta_b'],
                             gamma=s['gamma'])
    model.init_model(print_every=1, lr=0.01)
    model.train_model(iterations=5000, print_every=1000)
    
    alpha_a_nn, alpha_b_nn, beta_a_nn, beta_b_nn, gamma = model._get_best_params()
    
    s_new = {'S':5000000,
             'I_a':50,
             'I_b':50,
             'R':0,
             'D':0,
             
             'alpha_a':alpha_a_nn,
             'beta_a':beta_a_nn,
             
             'alpha_b':alpha_b_nn,
             'beta_b':beta_b_nn,
             
             'gamma':gamma
             }
    
    
    t_nn_param, wsol_nn_param = solver.solve_SIRD(s_new)
    
    solver.plot_SIRD(t, wsol, title='Train data')
    solver.plot_SIRD(t_nn_param, wsol_nn_param, title='Prediction')
    
    sirds = []
    for w in [wsol, wsol_nn_param]:
        S = w[:,0]
        I_a = w[:,1]
        I_b = w[:,2]
        I = I_a + I_b
        R = w[:,3]
        D = w[:,4]
        s = np.c_[S,I,I_a,I_b,R,D]
        sirds.append(s)
    
    fig, axes = plt.subplots(3,2)
    axes = axes.flatten()
    for i, label in enumerate(['S','I', 'I_a','I_b','R','D']):
        ax = axes[i]
        ax.plot(t, sirds[0][:,i], label='real')
        ax.plot(t_nn_param, sirds[1][:,i], label='pred')
        ax.set_title(label)
        ax.grid(linestyle=':') #
        ax.set_axisbelow(True)
    ax.legend()
    plt.tight_layout()
    
    dde.saveplot(model.losshistory, model.train_state, issave=False, isplot=True)
    
    