import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from .models import *

def costFunction(parameters, tSpan, Y0, data):
  sol = solve_ivp(lambda t, Y: covidModel(t, Y, parameters), tSpan, Y0, dense_output = True, atol = 1e-6, rtol = 1e-3, max_step = 0.1)
  t = np.linspace(tSpan[0], tSpan[1], len(data))
  Y = sol.sol(t).T
  loss = np.sum((Y - data) ** 2)
  return loss

def computeGradient(parameters, tSpan, Y0, data, tol = 1e-8):
  grad = np.zeros_like(parameters)
  for idx in range(len(parameters)):
    paramNeg = np.array(parameters, copy = True)
    paramPos = np.array(parameters, copy = True)
    paramNeg[idx] -= tol
    paramPos[idx] += tol
    costNeg = costFunction(paramNeg, tSpan, Y0, data)
    costPos = costFunction(paramPos, tSpan, Y0, data)
    grad[idx] = (costPos - costNeg) / (2 * tol)
  return grad

def gradientDescent(parameters, tSpan, Y0, data, lrate, epochs, paramBound):
  for idx in range(epochs):
    grad = computeGradient(parameters, tSpan, Y0, data, 1e-5)
    parameters -= lrate * grad
    parameters = np.clip(parameters, paramBound[0], paramBound[1])
    print(parameters)
    if (idx + 1) % 25 == 0:
      loss = costFunction(parameters, tSpan, Y0, data)
      print(f"Iteration - [{idx + 1}/{epochs}] Loss - {loss}")
  return parameters

def stochasticGradientDescent(parameters, tSpan, Y0, data, lrate, epochs, paramBound):
  pass