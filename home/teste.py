import numpy as np

def harmonic_oscillator(y, t, m, k):
    """
    Second-order differential equation for the harmonic oscillator.
    """
    dy = np.zeros_like(y)
    dy[0] = y[1]
    dy[1] = -(k/m) * y[0]
    return dy

def runge_kutta_2(y0, t, m, k, h):
    """
    Second-order Runge-Kutta method for solving the harmonic oscillator equation.
    """
    y = np.array(y0)
    for i in range(1, len(t)):
        k1 = h * harmonic_oscillator(y, t[i-1], m, k)
        k2 = h * harmonic_oscillator(y + k1, t[i], m, k)
        y = y + (k1 + k2) / 2
    return y

# Initial conditions
y0 = [1, 0]  # initial displacement and velocity
m = 1  # mass
k = 1  # spring constant

# Time steps
h = 0.01  # step size
t = np.arange(0, 10, h)  # time array

# Solve the equation
solution = runge_kutta_2(y0, t, m, k, h)

# Print the solution
print(solution)

import matplotlib.pyplot as plt

# Get the displacement and velocity arrays from the solution
displacement = solution[:, 0]
velocity = solution[:, 1]

# Plot the displacement in function of time
plt.plot(t, displacement, label='displacement')

# Plot the velocity in function of time
plt.plot(t, velocity, label='velocity')

# Add labels and a legend to the graph
plt.xlabel('Time (s)')
plt.ylabel('Y')
plt.legend()

# Show the plot
plt.show()