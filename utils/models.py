def covidModel(t, Y, parameters):
  T, I, V = Y
  lambda_, rho, beta, gamma, n, C = parameters
  dTdt = lambda_ - (rho * T) - (beta * T * V)
  dIdt = (beta * T * V) - (gamma * I)
  dVdt = (n * gamma * I) - (C * V) - (beta * T * V)
  return [dTdt, dIdt, dVdt]