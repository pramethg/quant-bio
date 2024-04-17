import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def dataTIV(tsteps):
  t = np.arange(1, tsteps + 1)
  T = 10000 - np.log(t) * 500 + np.random.normal(0, 120, tsteps)
  I = np.log(t) * 200 + np.random.normal(0, 80, tsteps)
  V = np.log(t) * 300 + np.random.normal(0, 150, tsteps)
  T = np.maximum(0, T)
  I = np.maximum(0, I)
  V = np.maximum(0, V)
  data = np.array([T, I, V])
  data = data.T
  return data

def dataExp(plot):
  data = pd.read_csv('../data/expCells.csv')
  data = data.to_numpy()
  if plot:
    titles = ['T', "logV"]
    plt.figure(figsize = (10, 4))
    for idx in range(2):
      plt.subplot(1, 2, idx + 1)
      plt.plot(data[:, 0], data[:, idx + 1], "r")
      plt.xlabel("Time")
      plt.ylabel("Population Count")
      plt.title(titles[idx])
    plt.suptitle("Experimental COVID Population Dynamics")
    plt.show()
  return data

def plot():
  data = dataTIV(10000)
  plt.figure(figsize = (6, 4))
  for idx in range(3):
    plt.plot(data[0], data[idx + 1])
  plt.legend(['T', 'I', 'V'])
  plt.show()

if __name__ == "__main__":
  data = dataExp(True)
  print(data.shape)