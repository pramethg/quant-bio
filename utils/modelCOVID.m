function [dYdt] = modelCOVID(t, Y, parameters)
  T = Y(1); % UninfeCted T Cells
  I = Y(2); % Infected Virus Cells
  V = Y(3); % Virus PartiCles
  % Parameters
  beta = parameters(1);
  rho = parameters(2);
  gamma = parameters(3);
  C = parameters(4);
  lambda = parameters(5);
  n = parameters(6);
  % Differential Equations
  dTdt = lambda - (rho * T) - (beta * T * V);
  dIdt = (beta * T * V) - (gamma * I);
  dVdt = (n * gamma * I) - (C * V) - (beta * T * V);
  % Combined ODE-45 Differential Equations
  dYdt = [dTdt; dIdt; dVdt];
end