function logPrior = priorFunction(parameters, bounds)
  logPrior = 0;
  lambda = parameters(1);
  rho = parameters(2);
  beta = parameters(3);
  gamma = parameters(4);
  n = parameters(5);
  C = parameters(6);
  for idx = 1:length(parameters)
    if parameters(idx) < bounds(1, idx) || parameters(idx) > bounds(2, idx)
      logPrior = -Inf;
      break;
    end
  end
end