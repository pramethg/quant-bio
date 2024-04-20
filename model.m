%%
Y0 = [50, 600, 50000];
tSpan = [0 60];
% Parameters : [lambda, rho, beta, gamma, n, c]
parameters = [80, 0.15, 0.00002, 0.55, 900, 5.5];
[t, Y] = ode45(@(t, Y) modelCOVID(t, Y, parameters), tSpan, Y0);

titles = ["T (Uninfected T Cells)", "I (Infected Virus Cells)", "logV (Virus Particles)"];
colors = ["b", "g", "r"];
figure;
for idx = 1:3
  subplot(1, 3, idx)
  if idx == 3
    plot(t, abs(log(Y(:, idx))), 'Color', colors(idx));
  else
    plot(t, Y(:, idx), 'Color', colors(idx));
  end
  title(titles(idx));
  xlabel('Time');
  ylabel('Population Count');
end
sgtitle('COVID Population Dynamics');

%%
data = readtable('./data/expCells.csv');
data = table2array(data);
initialParameters = [0.00002, 0.15, 0.55, 5.5, 80, 900];
nIters = 1e4;
stepSize = 3e-4;
[paramChain, acceptChain] = runMCMC(data, initialParameters, nIters, stepSize);
nParams = size(paramChain, 2);

fig = figure;
for idx = 1:nParams
  subplot(nParams, 1, idx);
  plot(paramChain(:, idx));
  xlabel('Iteration');
  ylabel(sprintf('Parameter %d', idx));
end