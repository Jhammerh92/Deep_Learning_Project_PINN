from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np

# class SIR:
#     def __init__(self):
#         pass

def vectorfield(w, t, p):
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
    alpha_a, alpha_b, beta_a, beta_b, gamma_a, gamma_b = p
    t_b = 25
    # Create f = (x1',y1',x2',y2'):
    f = [ - (alpha_a/(N) * I_a + alpha_b/(N) * I_b ) * S,
        (alpha_a/(N)) * S * I_a - beta_a *I_a - gamma_a*I_a,
        (alpha_b/(N)) * S * I_b - beta_b *I_b - gamma_b*I_b,
        beta_a * I_a + beta_b * I_b,
        gamma_a * I_a + gamma_b * I_b,
        ]
    return f


# Parameter values
# Masses:
alpha_a = 0.015
alpha_b = 0.02
beta_a = 0.001
beta_b = 0.001
gamma_a = 0.01
gamma_b = 0.001
# print(alpha/beta)

# Initial conditions
# x1 and x2 are the initial displacements; y1 and y2 are the initial velocities
I_a = 500000
I_b = 10
D = 500
R = 1000000
S = 5000000 - (I_a + I_b + D + R)

N = S

# ODE solver parametersq
abserr = 1.0e-8
relerr = 1.0e-6
stoptime = 365 *3
numpoints = 1000

# Create the time samples for the output of the ODE solver.
# I use a large number of points, only because I want to make
# a plot of the solution that looks nice.
# t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]
t = np.linspace(0, stoptime, numpoints, endpoint=True)
# print(t)
# Pack up the parameters and initial conditions:
p = [alpha_a, alpha_b, beta_a, beta_a, gamma_a, gamma_b]
w0 = [S, I_a, I_b, R, D]

# Call the ODE solver.
wsol = odeint(vectorfield, w0, t, args=(p,),
            atol=abserr, rtol=relerr)

# return wsol



# with open('two_springs.dat', 'w') as f:
#     # Print & save the solution.
#     for t1, w1 in zip(t, wsol):
#         print >> f, t1, w1[0], w1[1], w1[2], w1[3]

# for t1, w1 in zip(t, wsol):
#     print( t1, w1[0], w1[1], w1[2] )

print("total of each ",(wsol[-1,:]).astype(np.int32) )
print("total ",np.sum(wsol[-1,:]) )

plt.figure()
plt.plot(t, (wsol))
plt.legend(['S', 'I_a','I_b', 'R', 'D'])
plt.show()