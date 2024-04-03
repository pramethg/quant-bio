clc; clear; close all;

tstop = 100;
tspan = [0 tstop];
time = [];
Xi = [0 0];
X = zeros(length(Xi), tstop);
X(:, 1) = Xi;
C = [0.1 0.05];
H = updateCH(C, Xi);
Ch = C .* H';
i = 2;
t = 0;
colors = ["b", "g", "yellow", "cyan"];
labels = ["Time", "Concentration", "Error"];
tauArr = [];

%% Direct Method Solution
while (t < tstop)
  r1 = rand();
  r2 = rand();
  A = sum(Ch(:));
  tau = 1 / (A + 1e-5) * log(1 / (r1 + 1e-5));
  tau_arr(i - 1) = tau;
  mu = find((cumsum(Ch) >= r2 * A), 1, 'first');
  if isempty(mu)
    continue
  else
    X(:, i) = reaction(X(:, i - 1), mu);
    [H, Ch] = update_ch(C, X(:, i));
    i = i + 1;
    t = t + tau;
    time(i) = t;
  end
end

%% Direct Method Solution Plotting

%% ODE-45 Solution Calculation, Errors

%% ODE-45 Solution Plotting

%% Plot Analyzing Tau_Values With Respect to Time
figure;
plot(1:length(tauArr), tauArr, Color = "blue");
hold on;
plot(1:length(tauArr), movmean(tauArr, 10), Color = "green");
xlabel("Iteration #");
ylabel("Tau (a.u.)");
title(sprintf("Tau Value, Average: %d", mean(tauArr)));
xlim([0 length(tauArr) + 100]);
ylim([-0.5 1.5]);

%% Function for Drug Resistance Reactions
function Xv = reaction(Xv, mu)
  switch mu
    case 1
    case 2
    case 3
    case 4
  end
end

%% Function for Describing the ODE Behavior to the ODE-45 Solver
function dydt = drugResReaction(t, y, C)
end

%% Function to Update the C.H Matrix
function [H, Ch] = updateCH(C, X)
  H = X;
  Ch = C .* H';
end

%% Function to Plot Counts
function plotCounts(x, y, colors, labels, stitle)
  for i = 1:2
    plot(x, y(i, :), Color = colors(i));
    hold on;
  end
  xlabel(labels(1));
  ylabel(labels(2));
  title(stitle);
  legend(["R", "S"]);
  hold off;
end