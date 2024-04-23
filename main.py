import argparse
import numpy as np
import jax.numpy as jnp
from jax import grad, jit
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from utils import *

def argparser():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", type = str, default = "ode")
  parser.add_argument("--parameters", type = str, default = "80, 0.15, 0.00002, 0.55, 900, 5.5")
  parser.add_argument("--initial", type = str, default = "50, 600, 50000")
  parser.add_argument("--tspan", type = int, default = 60)
  parser.add_argument("--niters", type = int, default = 10000)
  parser.add_argument("--stepsize", type = float, default = 3e-5)
  parser.add_argument("--plot", action = 'store_true')
  parser.add_argument("--epochs", type = int, default = 300)
  parser.add_argument("--lrate", type = float, default = 3e-5)
  parser.add_argument("--rand", action = 'store_true')
  return parser

def implot(t, Y, data, legend = "ODE"):
  titles = ["T (Uninfected T Cells)", "I (Infected Virus Cells)", "logV (Virus Particles)"]
  colors = ["b", "g", "r"]
  plt.figure(figsize = (15, 5))
  for idx in range(3):
    plt.subplot(1, 3, idx + 1)
    if idx == 2:
      plt.plot(t, np.abs(np.log(Y[idx])), color = colors[idx])
    else:
      plt.plot(t, Y[idx], color = colors[idx])
    plt.scatter(data[:, 0], data[:, idx + 1], c = colors[idx])
    plt.title(titles[idx])
    plt.xlabel("Time")
    plt.ylabel("Population Count")
    plt.legend([legend, 'Experimental'])
  plt.suptitle("COVID Population Dynamics")
  plt.tight_layout()
  plt.show()

def ode(args):
  tSpan = (0, args.tspan)
  Y0 = [float(yidx) for yidx in (args.initial).split(",")]
  parameters = [float(sidx) for sidx in (args.parameters).split(",")]
  sol = solve_ivp(lambda t, Y: covidModel(t, Y, parameters), tSpan, Y0, dense_output = True)
  t = np.linspace(tSpan[0], tSpan[1], 300)
  Y = sol.sol(t)
  data = dataExp('./data/expCells.csv', 0)
  if args.plot:
    implot(t, Y, data)

def gradOptimizer(args):
  data = dataExp('./data/expCells.csv', 0)
  tSpan = (0, args.tspan)
  Y0 = [float(yidx) for yidx in (args.initial).split(",")]
  if args.rand:
    parameters = np.random.rand(6)
  else:
    parameters = [float(sidx) for sidx in (args.parameters).split(",")]
  if args.model == "ngd":
    parameters = gradientDescent(parameters, tSpan, Y0, data[:, 1:], args.lrate, args.epochs, args.bounds)
  sol = solve_ivp(lambda t, Y: covidModel(t, Y, parameters), tSpan, Y0, dense_output = True)
  t = np.linspace(tSpan[0], tSpan[1], 300)
  Y = sol.sol(t)
  if args.plot:
    implot(t, Y, data, "ODE")
  
def jaxOptimizer(args):
  data = dataExp('./data/expCells.csv', 0, 1)
  tSpan = (0, args.tspan)
  t = jnp.linspace(tSpan[0], tSpan[1], len(data))
  if args.initial == "0":
    Y0 = data[0, 1:]
  else:
    Y0 = jnp.array([float(yidx) for yidx in (args.initial).split(",")])
  if args.rand:
    parameters = jnp.array(np.random.rand(6))
  else:
    parameters = jnp.array([float(sidx) for sidx in (args.parameters).split(",")])
  if args.model == "jgd":
    parameters = gradientDescentJax(parameters, t, Y0, data[:, 1:], args.lrate, args.epochs)
  print(parameters.shape)

def mcmc(args):
  data = dataExp("./data/expCells.csv", 0)
  initParam = [0.00002, 0.15, 0.55, 5.5, 80, 900]
  paramNames = ['beta', 'rho', 'gamma', 'lambda', 'c', 'n']
  nParams = len(initParam)
  (paramChain, acceptChain) = runMCMC(data, initParam, args.niters, args.stepsize)
  plt.figure(figsize = (5, 20))
  for idx in range(nParams):
    plt.subplot(nParams, 1, idx + 1)
    plt.plot(paramChain[:, idx])
    plt.title(f"Parameter - {paramNames[idx].upper()}")
  plt.show()

def train(args):
  pass

if __name__ == "__main__":
  args = argparser().parse_args()
  args.bounds = [[0.000005, 0.05, 0.2, 2, 50, 750], [0.00006, 0.4, 1, 10, 120, 1200]]
  if args.model == "ode":
    ode(args)
  elif args.model == "mcmc":
    mcmc(args)
  elif args.model[0] == "j":
    if args.model in ["jgd", "jsgd", "jrmsprop", "jadam"]:
      jaxOptimizer(args)
  elif args.model in ["gd", "sgd", "rmsprop", "adam"]:
    gradOptimizer(args)