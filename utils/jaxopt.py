import numpy as np
import jax.numpy as jnp
from jax import grad, jit, device_get
from scipy.integrate import solve_ivp
from jax.experimental.ode import odeint
from jax.experimental.host_callback import id_print
from functools import partial

def covidModelJax(t, Y, parameters):
  T, I, V = Y
  lambda_, rho, beta, gamma, n, C = parameters
  dTdt = lambda_ - (rho * T) - (beta * T * V)
  dIdt = (beta * T * V) - (gamma * I)
  dVdt = (n * gamma * I) - (C * V) - (beta * T * V)
  return jnp.array([dTdt, dIdt, dVdt])

def inferParametersJax(parameters, Y0, t):
  sol = odeint(lambda Y, t: covidModelJax(t, Y, parameters), Y0, t)
  return sol

def lossFunctionJax(parameters, Y0, t, data):
  sol = odeint(lambda Y, t: covidModelJax(t, Y, parameters), Y0, t)
  return jnp.sum((sol - data) ** 2)

@partial(jit, static_argnums = (5, ))
def gradientDescentJax(parameters, t, Y0, data, lrate, epochs):
  gradCost = grad(lossFunctionJax)
  for idx in range(epochs):
    grads = gradCost(parameters, Y0, t, data)
    parameters -= lrate * grads
    loss = lossFunctionJax(parameters, Y0, t, data)
    if (idx + 1) % 25 == 0:
      print(f"Iteration - [{idx + 1} / {epochs}] Loss - {loss}")
  return parameters