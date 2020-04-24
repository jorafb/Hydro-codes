"""
Finite differencing example
Advection equation

@author: J. R. Fuentes (Jorafb)
Apr. 24th 2020
"""

import numpy as np
import matplotlib.pyplot as plt

# Setting up the grid and constant variables

Ngrid = 50
Nsteps = 3000
dt = 1
dx = 1
v = -0.1 
x = np.linspace(1,50,Ngrid)
count = 0

# Defining initial conditions

f1, f2 = np.copy(x)/Ngrid, np.copy(x)/Ngrid

# Setting up the plot

plt.ion()
fig, ax = plt.subplots(nrows=1,ncols=2)

ax[0].set_title('Forward-time central space')
ax[0].set_xlim([0,Ngrid])
ax[0].set_ylim([-1,2])
ax[0].set_xlabel(r'$x$')
ax[0].set_ylabel(r'$f$')

ax[1].set_title('Lax-Friedrichs')
ax[1].set_xlim([0,Ngrid])
ax[1].set_ylim([-1,2])
ax[1].set_xlabel(r'$x$')

# Setting initial state
ax[0].plot(x,f1,marker='o',linestyle='--',color='k')
ax[1].plot(x,f2,marker='o',linestyle='--',color='k')

# Plots that will be updated

ax1, = ax[0].plot(x,f1,marker='o',linestyle=' ',color='#1f77b4')
ax2, = ax[1].plot(x,f2,marker='o',linestyle=' ',color='#ff7f0e')

fig.canvas.draw()

plt.subplots_adjust(wspace=0.2)

# Time-evolution and updating plot

while count < Nsteps:
    
    # Forward-time central space method
    f1[1:Ngrid-1] =  f1[1:Ngrid-1] - (0.5*v*dt/dx)*(f1[2:Ngrid]-f1[0:Ngrid-2])
    ax1.set_ydata(f1)
    
    # Lax-Friedrichs method
    f2[1:Ngrid-1] =  0.5*(f2[2:Ngrid] + f2[0:Ngrid-2]) - (0.5*v*dt/dx)*(f2[2:Ngrid]-f2[0:Ngrid-2])
    ax2.set_ydata(f2)
    
    fig.canvas.draw()
    plt.pause(0.001)
    count += 1




