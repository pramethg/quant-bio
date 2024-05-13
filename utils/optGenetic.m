function [bestParameters, bestError] = optGenetic(tData, YData, paramBounds)
  function error = objectiveFunction(parameters)
    options = odeset('RelTol', 1e-6, 'AbsTol', 1e-9, 'MaxStep', 0.1);
    [~, YModel] = ode15s(@(t, Y) modelCOVID(t, Y, parameters), tData, YData(1, :), options);
    if isempty(YModel)
      error = inf;
    else
      error = sum(sum((YModel - YData).^2));
    end
  end
  opts = optimoptions(@ga, 'Display', 'iter', 'MaxGenerations', 100);
  [bestParameters, bestError] = ga(objectiveFunction, 6, [], [], [], [], paramBounds(1, :), paramBounds(2, :), [], opts);
end