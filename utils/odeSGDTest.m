function [parameters] = odeSGDTest(tData, YData, intialParameters, lrate, epochs, batchSize, delta)
  parameters = intialParameters;
  function error = costFunction(parameters, tBatch, YBatch)
    [~, YModel] = ode15s(@(t, Y) modelCOVID(t, Y, parameters), tBatch, YBatch(1, :));
    error = sum(sum((YModel - YBatch).^2));
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
      fprintf('Loss - %.2f\n', lossPos);
      return;
      if isnan(lossNeg) || isnan(lossPos)
        grad(pidx) = 0;
      else
        grad(pidx) = (lossPos - lossNeg) / (2 * delta);
      end
    end
    parameters = parameters - lrate * grad;
    return;
  end
end