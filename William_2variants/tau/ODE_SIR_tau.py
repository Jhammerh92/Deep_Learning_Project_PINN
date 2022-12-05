from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

class ODESolver:
    def __init__(self):
        pass
    
    def _SIRD_vectorfield(self, w, t, p):
        """
        Defines the differential equations for the coupled spring-mass system.
    
        Arguments:
            w :  vector of the state variables:
                      w = [x1,y1,x2,y2]
            t :  time
            p :  vector of the parameters:
                      p = [m1,m2,k1,k2,L1,L2,b1,b2]
        """
        S, I, R, D = w
        alpha, beta, gamma, N = p
    
        # Create f = (x1',y1',x2',y2'):
        f = [-(alpha/(N)) * S * I,
             (alpha/(N)) * S*I - beta *I - gamma*I,
             beta * I,
             gamma * I]
        return f
    
    def solve_SIRD(self, alpha=0.2, beta=0.05, gamma=0.01, init_num_people=5000000):
        # Initial conditions
        I = 10
        S = init_num_people - I
        R = 0
        D = 0
        self.init_num_people = init_num_people
        
        N = S
        
        # ODE solver parameters
        abserr = 1.0e-8
        relerr = 1.0e-6
        stoptime = 250.0
        numpoints = 250
        
        # Create the time samples for the output of the ODE solver.
        # I use a large number of points, only because I want to make
        # a plot of the solution that looks nice.
        # t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]
        t = np.linspace(0, stoptime, numpoints, endpoint=True)
        # Pack up the parameters and initial conditions:
        p = [alpha, beta, gamma, N]
        w0 = [S, I, R, D]
        
        # Call the ODE solver.
        wsol = odeint(self._SIRD_vectorfield, w0, t, args=(p,),
                      atol=abserr, rtol=relerr)
        return t, wsol, N
    
    def add_noise(self, array, loc=0.0, scale_pct=0.1):
        scale = array*scale_pct
        return array + np.random.normal(loc=loc,scale=scale, size = array.shape)
    
    def _axis_SIRD(self, ax):
        ax.set_xlabel('Time [day]')
        ax.set_ylabel('Number of people')
        
    def plot_SIRD(self, t, wsol, ax=None, title=None):
        print("total ",(wsol[-1,:]) )
        print("total ",np.sum(wsol[-1,:]) )
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(t, wsol)
        ax.legend(['S', 'I', 'R', 'D'])
        ax.grid()
        self._axis_SIRD(ax)
        if title is not None:
            ax.set_title(title)
        # plt.show()
        
    def plot_SIRD_scatter(self, t, wsol, ax=None, title=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.scatter(t, wsol[:,0],label='S')
        ax.scatter(t, wsol[:,1],label='I')
        ax.scatter(t, wsol[:,2],label='R')
        ax.scatter(t, wsol[:,3],label='D')
        ax.grid()
        ax.legend()
        self._axis_SIRD(ax)
        if title is not None:
            ax.set_title(title)
        # plt.show()
        
    def plot_synthetic_and_sample(self, t, wsol, t_sub, wsol_sub, ax=None, title=None):
        if ax is None:
            fig, ax = plt.subplots()
        
        linewidth = 3
        lineS = ax.plot(t, wsol[:,0], label='S', linewidth=linewidth)
        lineI = ax.plot(t, wsol[:,1], label='I', linewidth=linewidth)
        lineR = ax.plot(t, wsol[:,2], label='R', linewidth=linewidth)
        lineD = ax.plot(t, wsol[:,3], label='D', linewidth=linewidth)
        
        alpha=0.7
        ax.scatter(t_sub, wsol_sub[:,0],alpha=alpha)
        ax.scatter(t_sub, wsol_sub[:,1],alpha=alpha)
        ax.scatter(t_sub, wsol_sub[:,2],alpha=alpha)
        ax.scatter(t_sub, wsol_sub[:,3],alpha=alpha)
        
        red_circle = Line2D([0], [0], marker='o', color='w', label='Train data',
                        markerfacecolor='grey', markersize=8),
        
        ax.legend(handles=[lineS[0], lineI[0], lineR[0], lineD[0], red_circle[0]])
        
        self._axis_SIRD(ax)
        ax.grid(linestyle=':') #
        ax.set_axisbelow(True)
        
        if title is not None:
            ax.set_title(title)