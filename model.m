%%
initialize_v1;
data = readtable('./data/expCells.csv');
data = table2array(data);
Y0 = [50, 600, 50000];
YData = data(1, 2:end);
tSpan = [0 60];
tData = linspace(tSpan(1), tSpan(2), length(data));
paramNames = ["beta", "rho", "gamma", "c", "lambda", "n"];
initialParameters = [0.00002, 0.15, 0.55, 5.5, 80, 900];
expDataConv = data;
expDataConv(:, 4) = exp(expDataConv(:, 4));
addedExpData = zeros(size(expDataConv, 1) + 3, 4);
addedExpData(1:4, :) = expDataConv(1:4, :);
addedExpData(5, :) = mean(expDataConv(4:5, :));
addedExpData(6:14, :) = expDataConv(5:13, :);
addedExpData(15, :) = mean(expDataConv(13:14, :));
addedExpData(16:24, :) = expDataConv(14:22, :);
addedExpData(25, :) = mean(expDataConv(22:23, :));
addedExpData(26:end, :) = expDataConv(23:end, :);

%% ODE
[t, Y] = ode45(@(t, Y) modelCOVID(t, Y, initialParameters), tData, Y0);
odeplot(t, Y, data, "ODE");

%%
initialParameters = [0.00002, 0.15, 0.55, 5.5, 80, 900];
paramBounds = [0.1e-7, 0.0005, 0.003, 0.05, 1, 50; ...
               0.3e-3, 1, 20, 200, 10000, 15000];
nIters = 5e5;
stepSize = 1e-4;
[paramChain, acceptChain] = runMCMC(data, initialParameters, nIters, stepSize, paramBounds);
nParams = size(paramChain, 2);

fig = figure;
for idx = 1:nParams
  subplot(nParams, 1, idx);
  plot(paramChain(:, idx));
  xlabel('Iteration');
  ylabel(sprintf('%s', upper(initialParameters(idx))));
  title(sprintf('Parameter [%d] - %s', idx, upper(paramNames(idx))));
end

%% PLOT BURN-IN
nParams = size(paramChain, 2);
burnIn = round(0.5 * size(paramChain, 1));
nIters = size(paramChain, 1);
paramMean = mean(paramChain(burnIn + 1:end, :));
paramStd = std(paramChain(burnIn + 1:end, :));
paramPostBurnIn = paramChain(burnIn + 1:end, :);
figure;
for idx = 1:nParams
  subplot(nParams, 1, idx)
  hold on;
  plot(1:burnIn, paramChain(1:burnIn, idx), "red");
  plot(burnIn:nIters, paramChain(burnIn:end, idx), "g");
  yline(paramMean(idx), "b", 'LineWidth', 2);
  hold off;
  xlabel('Iteration');
  ylabel(sprintf('%s', upper(paramNames(idx))));
  title(sprintf('Parameter [%d] - %s - %.6f', idx, upper(paramNames(idx)), paramMean(idx)));
end

%% HISTOGRAM ANALYSIS
figure;
for idx = 1:nParams
  subplot(nParams, 1, idx);
  hold on;
  histogram(paramPostBurnIn(:, idx));
  xVals = linspace(paramBounds(1, idx), paramBounds(2, idx), 1e3);
  priorPdf = ones(1, length(xVals)) / (paramBounds(2, idx) - paramBounds(1, idx));
  plot(xVals, priorPdf, 'g--', 'Color', 'r', 'LineWidth', 2);
  xline(paramMean(idx), 'LineWidth', 2, 'Label', sprintf('Mean - %.6f', paramMean(idx)));
  title(sprintf('Parameters - %s', upper(paramNames(idx))));
  legend('Posterior', 'Prior', 'Mean Value');
  hold off;
end

%% PRINT RESULT
paramStats = table(paramNames', initialParameters', paramMean', paramStd', min(paramPostBurnIn)', max(paramPostBurnIn)', ...
  'VariableNames', {'Parameter', 'Inital', 'Mean', 'StdDev', 'Min', 'Max'});
disp(paramStats)

%% MCMC RESULT
[tModel, YModel] = ode45(@(t, Y) modelCOVID(t, Y, paramMean), tSpan, YData);
YModelBounds = zeros([size(YModel), 2]);
paramIds = [1 3 6];
for yidx = 1:size(YModel, 3)
  if yidx == 3
    YModelBounds(:, yidx, 1) = abs(log(YModel(:, yidx))) + 1.96 * paramStd(paramIds(yidx));
    YModelBounds(:, yidx, 2) = abs(log(YModel(:, yidx))) - 1.96 * paramStd(paramIds(yidx));
  else
    YModelBounds(:, yidx, 1) = YModel(:, yidx) + 1.96 * paramStd(paramIds(yidx));
    YModelBounds(:, yidx, 2) = YModel(:, yidx) - 1.96 * paramStd(paramIds(yidx));
  end
end
odeplot(tModel, YModel, data, "MCMC", YModelBounds);

%% SGD
[paramsSGD] = odeSGDTest(expDataConv(:, 1), expDataConv(:, 2:end), initialParameters, 3e-12, 300, 30, 1e-12);
[t, YSGD] = ode45(@(t, Y) modelCOVID(t, Y, paramsSGD), tData, Y0);
odeplot(t, YSGD, data, "ODE");

%% 
function [] = odeplot(t, Y, data, varargin)
  titles = ["T (Uninfected T Cells)", "I (Infected Virus Cells)", "logV (Virus Particles)"];
  colors = ["b", "g", "r"];
  fig = figure;
  set(fig, 'Position', [100, 100, 1200, 350]);
  if nargin > 4
    YModelBounds = varargin{2};
  end
  for idx = 1:3
    subplot(1, 3, idx)
    if idx == 3
      plot(t, abs(log(Y(:, idx))), 'Color', colors(idx));
    else
      plot(t, Y(:, idx), 'Color', colors(idx));
    end
    hold on;
    scatter(data(:, 1), data(:, idx + 1), 'o', 'filled', 'MarkerFaceColor', colors(idx));
    if nargin > 4
      fill([t; flipud(t)], [YModelBounds(:, idx, 1); flipud(YModelBounds(:, idx, 2))], colors(idx), 'FaceAlpha', 0.2, 'EdgeAlpha', 0);
      legend([varargin{1}, "Experimental", "95% CI"]);
    else
      legend([varargin{1}, "Experimental"]);
    end
    hold off;
    title(titles(idx));
    xlabel('Time');
    ylabel('Population Count');
  end
  sgtitle(sprintf('COVID Population Dynamics - %s', upper(varargin{1})))
end