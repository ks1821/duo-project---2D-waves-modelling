# By Advay Iyer 02025617 & Kuljeet Singh 02015414

# In this section I am importing all the libraries I will need

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# In this section I am setting the domain of solution and the discretised grid

Lx = 100  # length of puddle in x-direction
Ly = 100  # length of puddle in y-direction
T = 20 # total time of simulation

Nx = 200 # number of points in x-direction
Ny = 200  # number of points in y-direction
Nt = 100  # number of points in time

dx = Lx / Nx  # spatial step size in x-direction
dy = Ly / Ny  # spatial step size in y-direction
dt = T / Nt  # time step size

u = np.zeros((Nx, Ny, Nt))  # domain of solution

# In this section I am defining arrays I would need (if neeeded)

tvals = np.linspace(0,T,Nt) # Defining an array to loop through time values
xvals = np.linspace(0,Lx,Nx) # Defining an array to loop through x values in the grid
yvals = np.linspace(dy,Ly,Ny) # Defining an array to loop through y values in the grid


# In this section I am setting the boundary conditions/initial values

c = 1.75 # Wave speed
u[0, :, :] = 0  # left boundary
u[Nx-1, :, :] = 0  # right boundary
u[:, 0, :] = 0  # bottom boundary
u[:, Ny-1, :] = 0  # top boundary

dudt = np.zeros((Nx,Ny))    # initial du/dt conditions
dudt[int(Nx/2 - 1), int(Ny/2-1)] = -9.8  # Setting a perturbation at the centre of the grid
dudt[int(Nx/4 - 1), int(Ny/4-1)] = -9.8  # Setting a perturbation at the lower left corner of the grid
dudt[int(Nx*3/4 - 1), int(Ny*3/4-1)] = -9.8 # Setting a perturbation at the upper right corner of the grid
dudt[int(Nx*3/4 - 1), int(Ny/4-1)] = -9.8 # Setting a perturbation at the lower right corner of the grid
dudt[int(Nx/4 - 1), int(Ny*3/4-1)] = -9.8 # Setting a perturbation at the lower left corner of the grid

# In this section I am implementing the numerical method

i = 1
j = 1
for y in yvals[1:Ny-1]: # Looping through y values within boundary
    i = 1 # Reinitiallising i for each y loop
    for x in xvals[1:Nx-1]: # Looping through x values within boundary

        # Implementing du/dt boundary conditions to find u values at the first time step
        u[i,j,1] = ((dt * c/ dx)**2 * (u[i-1,j,0] + u[i+1,j,0] + u[i,j-1,0] + u[i,j+1,0] - 4 * u[i,j,0]))*0.5 + u[i,j,0] + dt * dudt[i,j]
        i = i + 1 # incrementing i
    j = j + 1 # incrementing j
p = 1 # setting time to the first intial time step
for t in tvals[1:Nt-1]: # looping through time domain after the first time step
    i = 1 # re - initiallising i and j variables for each time loop
    j = 1
    for y in yvals[1:Ny - 1]:  # Looping through y values in domain
        i = 1 # Re - initiallising i for each y loop
        for x in xvals[1:Nx - 1]: # Looping through x values in domain
            # Implementing numerical method to solve for u
            u[i, j, p+1] = ( (dt * c/ dx)**2 * (u[i - 1, j, p] + u[i + 1, j, p] + u[i, j - 1, p] + u[i, j + 1, p] - 4 * u[i, j, p]))  + 2* u[i, j, p] -u[i,j,p-1]
            i = i + 1 # Incrementing i
        j = j + 1 # Incrementing j
    p = p + 1 # Incrementing p, the step in time

# In this section I am showing the results

plt.imshow(u[:, :, -1], cmap='jet', origin='lower', extent=[0, Lx, 0, Ly]) # Creating 2D heat plot
plt.colorbar()
plt.xlabel('x') # Adding title and axis labels
plt.ylabel('y')
plt.title('Solution at t=' + str(T))
plt.show() # Display the 2d heatmap plot

grad_x, grad_y = np.gradient(u[:, :, -1]) #Taking gradient of u values
fig, ax = plt.subplots()
ax.streamplot(xvals, yvals, grad_x, grad_y) # Creating streamplot
ax.set_title('Streamplot of Wave Equation') # Adding title and axis labels
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show() # Displaying the streamline plot

fig = plt.figure(figsize=(8, 8)) # Creating a 3d figure
ax = fig.add_subplot(111, projection='3d')
x, y = np.meshgrid(xvals, yvals)
def update_plot(frame): # Animating wireframe plot
    ax.clear()
    ax.set_xlabel('X') # Setting axis titles
    ax.set_ylabel('Y')
    ax.set_zlabel('U')
    ax.set_title('Wave Equation Simulation') # Setting plot title
    ax.plot_wireframe(x, y, u[:, :, frame], cmap='viridis') # Setting each t value to a frame in animation
    ax.set_xlim(0, Lx) # Setting axis domains
    ax.set_ylim(0, Ly)
    ax.set_zlim(-1, 1) # Looping from first time step to last time step
ani = FuncAnimation(fig, update_plot, frames=Nt, interval=50)
plt.show() # Displaying 3d animated plot

# In this section I am celebrating
print('CW done: I deserve a good mark')