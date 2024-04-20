function logPrior = priorFunction(parameters)
  logPrior = 0;
  lambda = parameters(1);
  rho = parameters(2);
  beta = parameters(3);
  gamma = parameters(4);
  n = parameters(5);
  C = parameters(6);
  minValues = [0.000005, 0.05, 0.2, 2, 50, 750];
  maxValues = [0.00006, 0.4, 1, 10, 120, 1200];
  for idx = 1:length(parameters)
    if parameters(idx) < minValues(idx) || parameters(idx) > maxValues(idx)
      logPrior = -Inf;
      break;
    end
  end
end