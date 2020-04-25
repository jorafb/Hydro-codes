"""
Finite differencing example
Diffusion equation

@author: J. R. Fuentes (Jorafb)
Apr. 25th 2020
"""

import numpy as np
import matplotlib.pyplot as plt

# Setting up the grid and constant variables

Ngrid = 100
Nsteps = 1000
x = np.linspace(1,100,Ngrid)
alpha = 0.5 # alpha = (2*dt*D/dx) < 1 for stability in explicit method
count = 0
beta = 1 # beta = #(dt*D/dx^2)  stable for all dt in implicit method


# Defining initial conditions

# Step function with fixed boundaries
v = np.zeros(Ngrid)
v[0] = 0
v[40:60] = 1
v[-1] = 0

# Setting up the plot
plt.ion()
fig, ax = plt.subplots(nrows=1,ncols=1)

#ax.set_title('Explicit')
ax.set_xlim([0,Ngrid])
ax.set_ylim([0,2])
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$v$')

# Setting initial state
ax.plot(x,v,marker=' ',linestyle='-',color='k',label='Initial condition')
ax.legend(frameon=None)
# Plot that will be updated
ax1, = ax.plot(x,v,marker=' ',linestyle='-',color='#1f77b4')

fig.canvas.draw()

# Turn on/off method

explicit = False
semi_implicit = True
implicit = False

# Time-evolution and updating plot

while count < Nsteps:
    
    if explicit == True:
        v[1:Ngrid-1] =  v[1:Ngrid-1] + alpha*(v[2:Ngrid]-2*v[1:Ngrid-1] + v[0:Ngrid-2])
    
    if semi_implicit == True:
            
        # Creating the tri-diagonal matrix for v_n+1
        A = -beta*np.eye(Ngrid,k=-1) + (2+2*beta)*np.eye(Ngrid,k=0) - beta*np.eye(Ngrid,k=1)
        # Boundary condition: (v[0] has to be fixed at the speed of the boundary)
        A[0][0] = 1
        A[0][1] = 0
        # Boundary condition: zero-gradient at right boundary
        A[-1][-1] = 1 + beta 
        
        # Creating the tri-diagonal matrix for v_n
        B = beta*np.eye(Ngrid,k=-1) + (2-2*beta)*np.eye(Ngrid,k=0) + beta*np.eye(Ngrid,k=1)
        # Boundary condition: (v[0] has to be fixed at the speed of the boundary)
        B[0][0] = 1
        B[0][1] = 0
        # Boundary condition: zero-gradient at right boundary
        B[-1][-1] = 1 + beta
        
        v[1:Ngrid-1] =  np.linalg.solve(A,np.dot(B,v))[1:Ngrid-1]
        
    
    if implicit == True:
        # Creating the tri-diagonal matrix for v_n+1
        A = -beta*np.eye(Ngrid,k=-1) + (1+2*beta)*np.eye(Ngrid,k=0) - beta*np.eye(Ngrid,k=1)
        # Boundary condition: (v[0] has to be fixed at the speed of the boundary)
        A[0][0] = 1
        A[0][1] = 0
        # Boundary condition: zero-gradient at right boundary
        A[-1][-1] = 1 + beta 
        
        v[1:Ngrid-1] = np.linalg.solve(A,v)[1:Ngrid-1]
        

    ax1.set_ydata(v)
    fig.canvas.draw()
    plt.pause(0.001)
    count += 1




