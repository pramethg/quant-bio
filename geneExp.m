clc; clear; close all;
addpath(genpath("./src/"));

tstop = 8 * 60 * 60;
tspan = [0 tstop];
time = []; % Array to store the Time Values
time(1) = tspan(1); % Initializing with 0
Xi = [0 0]; % Iniitial Concentration Vector for mRNA and Protein
X = zeros(length(Xi), tstop); % Counts Matrix Zero-Initialization
X(:, 1) = Xi; % Initializing the First Time Step with Initial Counts
b = 10; % Burst Size: b = k_p / gamma_r
k_r = 0.01; % Transcription Rate
gamma_r = 0.1; % mRNA Decay Rate
gamma_p = 0.002; % Protein Decay Rate
k_p = b * gamma_r; % Translation Rate
C = [k_r k_p gamma_r gamma_p]; % Rate Constants Vector for Transcription, Translation, mRNA Decay, Protein Decay
H = update_ch(C, Xi);
Ch = C.*H'; % Compute the Initial Ch Vector
i = 2; % Iteration Variable
t = 0; % Time Variable used for deciding the stopping condition
tau_arr = []; % Vector to store Tau-values
colors = ["blue", "magenta", "red", "cyan"];
labels = ["Time", "Number of mRNA/Protein", "Error"];

%% Main Loop for Direct Method Solution
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
%% ODE-45 Solver Solution
[sGeneTime, sGeneConc] = ode45(@(t, y) geneExpression(t, y, C), time, Xi);
% Error w.r.t. Direct Method and ODE-45 Solution
error = X' - sGeneConc(1:end - 1, :);

%% Plot the Direct Method Solution
figure;
plot(time(:, 1:end - 1), X(1, :), "Color", "blue");
hold on
plot(time(:, 1:end - 1), X(2, :), "Color", "green");
legend(["#mRNA", "#Protein"]);
xlabel("Time");
ylabel("Number of mRNA/Protein");
ylim([0, 120]);
title("Direct Method");

%% Plot Analyzing Differences Betweem Direct Method and ODE-45 Solution
figure;
subplot(2, 2, 1)
matPlot(time(:, 1:end - 1), X, colors(1:2), labels(1:2), "Direct Method Solution", ["#mRNA", "#Protein"], [0, 120]);
subplot(2, 2, 2)
matPlot(sGeneTime, sGeneConc', colors(1:2), labels(1:2), "ODE45 Solution", ["#mRNA", "#Protein"], [0, 120]);
subplot(2, 2, 3)
matPlot(time(:, 1:end - 1), cat(1, X(2, :), sGeneConc(1:end - 1, 2)'), colors([3, 1]), ["Time", "Protein Number"], "Direct Method vs. ODE-45 Solution", ["Direct Method", "ODE-45"], [0, 120]);
subplot(2, 2, 4)
matPlot(time(:, 1:end - 1), error', colors([3, 1]), labels([1, 3]), strcat("Sum of Errors(ODE45): ", num2str(sum(abs(error(:))))));

%% Plot Analyzing Tau-Value With Respect to Time
figure;
plot(1:length(tau_arr), tau_arr, Color = "blue");
hold on;
plot(1:length(tau_arr), movmean(tau_arr, 10), Color = "green");
xlabel("Iteration #");
ylabel("Tau (a.u.)");
title(strcat("Tau Value, Average: ", num2str(mean(tau_arr))));
xlim([0 length(tau_arr) + 20]);
legend(["Tau Value", "Moving Averaged(10) Tau Value"])

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