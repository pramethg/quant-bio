function [dYdt] = modelCOVID(t, Y, parameters)
  T = Y(1); % UninfeCted T Cells
  I = Y(2); % Infected Virus Cells
  V = Y(3); % Virus PartiCles
  % Parameters
  lambda = parameters(1);
  rho = parameters(2);
  beta = parameters(3);
  gamma = parameters(4);
  n = parameters(5);
  C = parameters(6);
  % Differential Equations
  dTdt = lambda - (rho * T) - (beta * T * V);
  dIdt = (beta * T * V) - (gamma * I);
  dVdt = (n * gamma * I) - (C * V) - (beta * T * V);
  % Combined ODE-45 Differential Equations
  dYdt = [dTdt; dIdt; dVdt];
end