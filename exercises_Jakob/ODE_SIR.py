from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np

raise Exception('Script has been moved to DINN_implementation folder :)')

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
    S, I, R, D = w
    alpha, beta, gamma = p

    # Create f = (x1',y1',x2',y2'):
    f = [-(alpha/(N)) * S * I,
         (alpha/(N)) * S*I - beta *I - gamma*I,
         beta * I,
         gamma * I]
    return f


# Parameter values
# Masses:
alpha = 0.2
beta = 0.05
gamma = 0.01
print(alpha/beta)

# Initial conditions
# x1 and x2 are the initial displacements; y1 and y2 are the initial velocities
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
print(t)
# Pack up the parameters and initial conditions:
p = [alpha, beta, gamma]
w0 = [S, I, R, D]

# Call the ODE solver.
wsol = odeint(vectorfield, w0, t, args=(p,),
              atol=abserr, rtol=relerr)

# with open('two_springs.dat', 'w') as f:
#     # Print & save the solution.
#     for t1, w1 in zip(t, wsol):
#         print >> f, t1, w1[0], w1[1], w1[2], w1[3]

# for t1, w1 in zip(t, wsol):
#     print( t1, w1[0], w1[1], w1[2] )

print("total ",(wsol[-1,:]) )
print("total ",np.sum(wsol[-1,:]) )

plt.figure()
plt.plot(t, wsol)
plt.legend(['S', 'I', 'R', 'D'])
plt.show()