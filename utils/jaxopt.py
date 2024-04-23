import jax.numpy as jnp
from jax import grad, jit
from scipy.integrate import solve_ivp

def covidModel(t, Y, parameters):
  T, I, V = Y
  lambda_, rho, beta, gamma, n, C = parameters
  dTdt = lambda_ - (rho * T) - (beta * T * V)
  dIdt = (beta * T * V) - (gamma * I)
  dVdt = (n * gamma * I) - (C * V) - (beta * T * V)
  return jnp.array([dTdt, dIdt, dVdt])

def lossFunction(parameters, Y0, tSpan, t, data):
  sol = solve_ivp(covidModel, tSpan, Y0, args = (parameters, ), t_eval = t)
  pred = sol.y.T
  return jnp.mean(jnp.sum((pred - data) ** 2, axis = 1))

@jit
def gradientDescentOpt(parameters, tSpan, t, Y0, data, lrate, epochs):
  for idx in range(epochs):
    grads = grad(lossFunction)(parameters, Y0, tSpan, t, data)
    parameters -= lrate * grads
    loss = lossFunction(parameters, Y0, tSpan, t, data)
    if (idx + 1) % 25 == 0:
      print(f"Iteration - [{idx + 1} / {epochs}] Loss - {loss}")
    else:
      print("Yes")