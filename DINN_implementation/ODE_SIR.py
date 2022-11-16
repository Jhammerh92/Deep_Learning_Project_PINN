from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np


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
    
    def solve_SIRD(self, alpha=0.2, beta=0.05, gamma=0.01):
        # Initial conditions
        I = 10
        S = 5000000 - I
        R = 0
        D = 0
        
        N = S
        
        # ODE solver parameters
        abserr = 1.0e-8
        relerr = 1.0e-6
        stoptime = 250.0
        numpoints = 1001
        
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
    
    def plot_SIRD(self, t, wsol):
        print("total ",(wsol[-1,:]) )
        print("total ",np.sum(wsol[-1,:]) )
        
        plt.figure()
        plt.plot(t, wsol)
        plt.legend(['S', 'I', 'R', 'D'])
        plt.show()