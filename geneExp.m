clc; clear; close all;
addpath(genpath("./src/"));

tstop = 8 * 60 * 60;
tspan = [0 tstop];
time = []; % Array to store the Time Values
time(1) = tspan(1); % Initializing with 0
Xi = [0 0]; % Iniitial Concentration Vector for mRNA and Protein
X_Values = {}; % Cell to Store Count Matrices, 2 Matrices for 2 Different b values.
T_Values = {}; % Cell to Store Time Matrices
b = [10, 1]; % Burst Size: b = k_p / gamma_r
k_r = [0.01 0.1]; % Transcription Rates
gamma_r = 0.1; % mRNA Decay Rate
gamma_p = 0.002; % Protein Decay Rate
k_p = b * gamma_r; % Translation Rate
colors = ["blue", "magenta", "red", "cyan"];
labels = ["Time (Hours)", "Number of mRNA/Protein", "Error"];

%% Main Loop for Direct Method Solution
for idx = 1:2
  % Initializing Variables for Direct Method Simulation
  i = 2; % Iteration Variable
  t = 0; % Time Variable used for deciding the stopping condition
  C = [k_r(idx) k_p(idx) gamma_r gamma_p]; % Rate Constants Vector for Transcription, Translation, mRNA Decay, Protein Decay
  H = update_ch(C, Xi);
  Ch = C .* H'; % Compute the Initial Ch Vector
  tau_arr = []; % Vector to store Tau-values
  X = zeros(length(Xi), tstop); % Zero-Initialization of Counts Matrix
  X(:, 1) = Xi; % Initializing the First Time Step with Initial Counts

  % Loop for Direct Method Calculation
  while (t < tstop)
    % Generate Random Real Numbers(Between 0 and 1)
    r1 = rand();
    r2 = rand();
    A = sum(Ch(:));
    % Compute Tau Value
    tau = 1 / (A + 1e-5) * log(1 / (r1 + 1e-5));
    % Store Tau Value
    tau_arr(i - 1) = tau;
    % Compute Mu Value
    mu = find((cumsum(Ch) >= r2 * A), 1, 'first');
    % If suitable Mu is not found, just continue with another set of Tau
    if isempty(mu)
      continue
    else
      % Carry out the Reaction Based on Mu, Update Counts
      X(:, i) = reaction(X(:, i - 1), mu);
      % Update the Ch Matrix
      [H, Ch] = update_ch(C, X(:, i));
      i = i + 1;
      t = t + tau;
      time(i) = t;
    end
  end
  X = X(:, 1:size(time, 2) - 1);
  X_Values{idx} = X;
  T_Values{idx} = time(1: end - 1) / 3600; % Convert to Hours
end

%% ODE-45 Solver Solution
odeCountValues = {};
odeTimeValues = {};
for idx = 1:2
  C = [k_r(idx) k_p(idx) gamma_r gamma_p]; % Rate Constants Vector for Transcription, Translation, mRNA Decay, Protein Decay
  % ODE-45 Solution
  [sGeneTime, sGeneConc] = ode45(@(t, y) geneExpression(t, y, C), linspace(0, max(T_Values{idx}(:)) * 3600, size(T_Values{idx}, 2)), Xi);
  % Error w.r.t. Direct Method and ODE-45 Solution
  error = X_Values{idx}' - sGeneConc;
  odeCountValues{idx} = sGeneConc;
  odeTimeValues{idx} = sGeneTime;
end

%% Plot the Direct Method Solution
figure(1);
subplot(1, 2, 1)
matPlot(T_Values{1}, X_Values{1}, colors(1:2), labels(1:2), "Direct Method Solution (b = 10)", ["#mRNA", "#Protein"], [0, 150]);
subplot(1, 2, 2)
matPlot(T_Values{2}, X_Values{2}, colors(1:2), labels(1:2), "Direct Method Solution (b = 1)", ["#mRNA", "#Protein"], [0, 150]);

%% Plot Analyzing Differences Betweem Direct Method and ODE-45 Solution
clc;
figure;
subplot(2, 2, 1)
matPlot(T_Values{1}, cat(1, X_Values{1}(2, :), odeCountValues{1}(:, 2)'), colors([3, 1]), ["Time", "Protein Number"], "Direct Method vs. ODE-45 Solution (b = 10)", ["Direct Method", "ODE-45"], [0, 150]);
subplot(2, 2, 2)
matPlot(T_Values{2}, cat(1, X_Values{2}(2, :), odeCountValues{2}(:, 2)'), colors([3, 1]), ["Time", "Protein Number"], "Direct Method vs. ODE-45 Solution (b = 1)", ["Direct Method", "ODE-45"], [0, 150]);
subplot(2, 2, 3)
plotHistogram(X_Values{1}(2, :), "Histogram of Protein Number (b = 10)");
subplot(2, 2, 4)
plotHistogram(X_Values{2}(2, :), "Histogram of Protein Number (b = 1)");

%% Function for Gene Expression Reactions
function Xv = reaction(Xv, mu)
  switch mu
    case 1
      Xv(1) = Xv(1) + 1; % Transcription: Produce mRNA
    case 2
      Xv(2) = Xv(2) + 1; % Translation: Produce Protein
    case 3
      Xv(1) = Xv(1) - 1; % mRNA Decay
    case 4
      Xv(2) = Xv(2) - 1; % Protein Decay
  end
end

%% Function to Describing the ODE Behavior to the ODE-45 Solver
function dydt = geneExpression(t, y, C)
  dmRNAdt = C(1) - C(3) * y(1);
  dProteindt = C(2) * y(1) - C(4) * y(2);
  dydt = [dmRNAdt; dProteindt];
end

%% Function to Update the C.H Matrix
function [H, Ch] = update_ch(C, X)
  H = [1; X(1); X(1); X(2)];
  Ch = C .* H';
end

%% Plotting Function to Plot Counts/Numbers
function matPlot(x, y, colors, labels, stitle, varargin)
  for i = 1:2
    plot(x, y(i, :), Color = colors(i));
    hold on;
  end
  xlabel(labels(1));
  ylabel(labels(2));
  title(stitle);
  if nargin > 5
    legend(varargin{1});
    ylim(varargin{2});
  else
    legend(["#mRNA", "#Protein"]);
  end
  hold off;
end

%% Plotting Histograms
function plotHistogram(X, stitle)
  indices = X > 10;
  filteredCounts = X(indices);
  histogram(filteredCounts, 'FaceColor', 'red');
  xlabel('Protein Number Bins');
  ylabel('Counts');
  title(stitle);
end
