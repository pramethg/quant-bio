import numpy as np
import pandas as pd
import jax.numpy as jnp
import matplotlib.pyplot as plt

def dataExp(fpath = '../data/expCells.csv', plot = 0, jaxarr = 0):
  data = pd.read_csv(fpath)
  data = data.to_numpy()
  if jaxarr:
    data = jnp.array(data)
  if plot:
    titles = ['T Cells', 'I Cells', 'logV Cells']
    colors = ['r', 'b', 'g']
    plt.figure(figsize = (10, 4))
    for idx in range(3):
      plt.subplot(1, 3, idx + 1)
      plt.plot(data[:, 0], data[:, idx + 1], colors[idx])
      plt.xlabel("Time")
      plt.ylabel("Population Count")
      plt.title(titles[idx])
    plt.suptitle("Experimental COVID Population Dynamics")
    plt.show()
  return data

if __name__ == "__main__":
  data = dataExp(fpath = '../data/expCells.csv', plot = 1)