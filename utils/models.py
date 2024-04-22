import numpy as np
from scipy.integrate import solve_ivp

def covidModel(t, Y, parameters):
  T, I, V = Y
  lambda_, rho, beta, gamma, n, C = parameters
  dTdt = lambda_ - (rho * T) - (beta * T * V)
  dIdt = (beta * T * V) - (gamma * I)
  dVdt = (n * gamma * I) - (C * V) - (beta * T * V)
  return [dTdt, dIdt, dVdt]

def priorFunction(parameters):
  logPrior = 0
  minValues = [0.00001, 0.1, 0.35, 3.5, 70, 750]
  maxValues = [0.00003, 0.2, 0.75, 7.5, 90, 1200]
  for idx in range(len(parameters)):
    if (parameters[idx] < minValues[idx]) or (parameters[idx] > maxValues[idx]):
      logPrior = -float("inf")
      break
  return logPrior

def logLikelihood(data, parameters):
  sigmaT, sigmaI, sigmalogV = 4.5, 0.5, 0.45
  time = data[:, 0]
  Y0 = data[0, 1:]
  sol = solve_ivp(lambda t, Y: covidModel(t, Y, parameters), (time[0], time[-1]), Y0, t_eval = time)
  Y = sol.y
  TModel = Y[0, :]
  IModel = Y[1, :]
  VModel = Y[2, :]
  resT = data[:, 1] - TModel
  resI = data[:, 2] - IModel
  resV = data[:, 3] - VModel
  logLikT = -0.5 * np.sum((resT / sigmaT) ** 2)
  logLikI = -0.5 * np.sum((resI / sigmaI) ** 2)
  logLikV = -0.5 * np.sum((resV / sigmalogV) ** 2)
  totalLogLik = (logLikT + logLikI + logLikV)
  return totalLogLik

def runMCMC(data, initialParams, nIters, stepSize):
  nParams = len(initialParams)
  paramsChain = np.zeros((nIters, nParams))
  acceptChain = np.zeros(nIters, dtype = bool)
  paramsChain[0, :] = initialParams
  currentLogLik = logLikelihood(data, initialParams)
  currentLogPrior = priorFunction(initialParams)
  for idx in range(1, nIters):
    paramProposal = paramsChain[idx - 1, :] + np.random.normal(0, stepSize, nParams)
    propLogLik = logLikelihood(data, paramProposal)
    propLogPrior = priorFunction(paramProposal)
    acceptProb = np.exp(propLogLik + propLogLik - currentLogLik - currentLogLik)
    if np.random.rand() <= acceptProb:
      paramsChain[idx, :] = paramProposal
      currentLogLik = propLogLik
      currentLogPrior = propLogPrior
      acceptChain[idx] = True
    else:
      paramsChain[idx, :] = paramsChain[idx - 1, :]
    if idx % 10 == 0:
      print(f"Iterations - [{idx} / {nIters}]")
  print(f"Completed MCMC Iterations - [{nIters} / {nIters}]")
  return paramsChain, acceptChain