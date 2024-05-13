function [parameters] = optSGD(tData, YData, intialParameters, lrate, epochs, batchSize, delta)
  parameters = intialParameters;
  function error = costFunction(parameters, tBatch, YBatch)
    options = odeset('RelTol', 1e-6, 'AbsTol', 1e-8, 'NonNegative', 1:length(parameters));
    try
      [~, YModel] = ode15s(@(t, Y) modelCOVID(t, Y, parameters), tBatch, YBatch(1, :), options);
      disp(size(YModel));
      error = sum(sum((YModel - YBatch).^2));
    catch
      error = inf;
    end
  end
  for iter = 1:epochs
    grad = zeros(size(parameters));
    for pidx = 1:length(parameters)
      paramPos = parameters;
      paramPos(pidx) = parameters(pidx) + delta;
      lossPos = costFunction(paramPos, tData, YData);
      paramNeg = parameters;
      paramNeg(pidx) = parameters(pidx) - delta;
      lossNeg = costFunction(paramNeg, tData, YData);
      if isnan(lossNeg) || isnan(lossPos)
        grad(pidx) = 0;
      else
        grad(pidx) = (lossPos - lossNeg) / (2 * delta);
      end
    end
    parameters = parameters - lrate * grad;
    if mod(iter, 100) == 0
      loss = costFunction(parameters, tData, YData);
      fprintf('Iteration - [%d / %d] Loss - %.2f\n', iter, epochs, loss);
    end
  end
end