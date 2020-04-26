"""
Finite differencing example
Advection-Diffusion equation

@author: J. R. Fuentes (Jorafb)
Apr. 26th 2020
"""

import numpy as np
import matplotlib.pyplot as plt

# Setting up the grid and constant variables

Ngrid = 100
Nsteps = 3000
x = np.linspace(1,100,Ngrid)
count = 0
dt=1
dx=1
D1 = 1
beta1 = dt*D1/(dx**2)
D2 = 0.1
beta2 = dt*D2/(dx**2)
v = -0.1

# Defining initial conditions

f1, f2 = np.copy(x)/Ngrid, np.copy(x)/Ngrid

# Setting up the plot
plt.ion()
fig, ax = plt.subplots(nrows=1,ncols=2)

ax[0].set_title(r'$D=1$')
ax[0].set_xlim([0,Ngrid])
ax[0].set_ylim([0,1.5])
ax[0].set_xlabel(r'$x$')
ax[0].set_ylabel(r'$f$')

ax[1].set_title(r'$D=0.1$')
ax[1].set_xlim([0,Ngrid])
ax[1].set_ylim([0,1.5])
ax[1].set_xlabel(r'$x$')

# Setting initial state
ax[0].plot(x,f1,marker=' ',linestyle='-',color='k',label=r'Initial Condition')
ax[1].plot(x,f2,marker=' ',linestyle='-',color='k',label=r'Initial Condition')
ax[0].legend()
ax[1].legend()

# Plot that will be updated
ax1, = ax[0].plot(x,f1,marker=' ',linestyle='-',color='#1f77b4')
ax2, = ax[1].plot(x,f2,marker=' ',linestyle='-',color='#ff7f0e')

fig.canvas.draw()

# Time-evolution and updating plot

while count < Nsteps:
    
    # Updating diffusion part
    # Creating the tri-diagonal matrix for diffusion 
    A1 = -beta1*np.eye(Ngrid,k=-1) + (1+2*beta1)*np.eye(Ngrid,k=0) - beta1*np.eye(Ngrid,k=1)
    A2 = -beta2*np.eye(Ngrid,k=-1) + (1+2*beta2)*np.eye(Ngrid,k=0) - beta2*np.eye(Ngrid,k=1)
    
    # Boundary condition: (f fixed at the first cell)
    A1[0][0] = 1
    A1[0][1] = 0
    
    A2[0][0] = 1
    A2[0][1] = 0
    
    # Boundary condition: zero-gradient at last cell
    A1[-1][-1] = 1 + beta1
    A2[-1][-1] = 1 + beta2 

    
    # Updating advection part
    f1[1:Ngrid-1] = np.linalg.solve(A1,f1)[1:Ngrid-1]
    f1[1:Ngrid-1] =  0.5*(f1[2:Ngrid] + f1[0:Ngrid-2]) - (0.5*v*dt/dx)*(f1[2:Ngrid]-f1[0:Ngrid-2])
    
    f2[1:Ngrid-1] = np.linalg.solve(A2,f2)[1:Ngrid-1]
    f2[1:Ngrid-1] =  0.5*(f2[2:Ngrid] + f2[0:Ngrid-2]) - (0.5*v*dt/dx)*(f2[2:Ngrid]-f2[0:Ngrid-2])
    
    ax1.set_ydata(f1)
    ax2.set_ydata(f2)
    
    fig.canvas.draw()
    plt.pause(0.001)
    count += 1




