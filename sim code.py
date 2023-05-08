###@piyawat luknatin 2023
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def sei_model(t, y, beta, alpha, gamma, delta, A, w):
    S, E, I = y
    dSdt = -beta * S * I * delta
    dEdt = beta * S * I * delta - alpha * E
    dIdt = alpha * E - gamma * I
    return [dSdt, dEdt, dIdt]

beta = 0.1
alpha = 0.1
gamma = 0.05
delta = 0.5

# Generate random adjacency matrix
n = 10
p = 0.12
A = np.random.binomial(1, p, size=(n, n))

# Ensure diagonal elements are 0
np.fill_diagonal(A, 0)

# Calculate average connectivity
K = p * (n - 1)
print(f"Average connectivity: {K}")

# Generate random weights
w = np.random.rand(n)

# Initial conditions
S0 = 1
E0 = 0.5
I0 = 0.5

# Time points
t = np.linspace(0, 100, 1000)

# Solve SEIW model
sol = solve_ivp(sei_model, [0, 100], [S0, E0, I0], args=(beta, alpha, gamma, delta, A, w), t_eval=t)

peak_I = np.max(sol.y[2])
print("Peak value of I:", peak_I)

# Create figure and axis objects
fig, ax = plt.subplots()
ax.set_xlim(0, 100)
ax.set_ylim(0, 1)
ax.set_xlabel('Time')
ax.set_ylabel('Fraction of population')
ax.set_title('SEI model with Erdős-Rényi network (delta = 1, beta = 0.1)')

# Plot results
ax.plot(sol.t, sol.y[0], label='Susceptible')
ax.plot(sol.t, sol.y[1], label='Exposed')
ax.plot(sol.t, sol.y[2], label='Infectious')
ax.legend()

plt.show()
