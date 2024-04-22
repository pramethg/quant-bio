import argparse
import numpy as np
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
  parser.add_argument("--stepsize", type = float, default = 3e-4)
  return parser

def ode(args):
  tspan = (0, args.tspan)
  Y0 = [float(yidx) for yidx in (args.initial).split(",")]
  parameters = [float(sidx) for sidx in (args.parameters).split(",")]
  sol = solve_ivp(lambda t, Y: covidModel(t, Y, parameters), tspan, Y0, dense_output = True)
  t = np.linspace(tspan[0], tspan[1], 300)
  Y = sol.sol(t)
  data = dataExp('./data/expCells.csv', 0)
  titles = ["T (Uninfected T Cells)", "I (Infected Virus Cells)", "logV (Virus Particles)"]
  colors = ["b", "g", "r"]
  plt.figure(figsize = (15, 5))
  for idx in range(3):
    plt.subplot(1, 3, idx + 1)
    if idx == 2:
      plt.plot(t, np.abs(np.log(Y[idx])), color = colors[idx])
    else:
      plt.plot(t, Y[idx], color = colors[idx])
    plt.scatter(data[:, 0], data[:, idx], c = colors[idx])
    plt.title(titles[idx])
    plt.xlabel("Time")
    plt.ylabel("Population Count")
    plt.legend(['ODE', 'Experimental'])
  plt.suptitle("COVID Population Dynamics")
  plt.tight_layout()
  plt.show()

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
  if args.model == "ode":
    ode(args)
  elif args.model == "mcmc":
    mcmc(args)
  elif args.model == "":
    train(args)