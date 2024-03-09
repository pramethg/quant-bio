%% Initializing Parameters
clc; clear; close all;

tstop = 100;
tspan = [0 tstop];
time = []; % Array to store the Time Values
time(1) = tspan(1); % Initializing with 0
Xi = [100 0]; % Iniitial Concentration Vector
X = zeros(length(Xi), tstop); % Counts Matrix Zero-Initialization
X(:, 1) = Xi; % Initializing the First Time Step with Initial Counts
C = [0.1 0.1]; % Rate Constants Vector
H = update_ch(C, Xi); % Using the C*H Update Function to Initialize the H Vector
Ch = C.*H'; % Compute the Initial Ch Vector
i = 2; % Iteration Variable
t = 0; % Time Variable used for deciding the stopping condition
colors = ["b", "g", "yellow", "cyan"];
labels = ["Time", "Concentration", "Error"];
tau_arr = []; % Vector to store Tau-values

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

%% Closed Form Solution, ODE-45 Solution and Errors
% Closed Form Solution
cfs(1, :) = Xi(1) * (C(2) + C(1)*exp(-(C(1) + C(2))*time)) / (C(1) + C(2));
cfs(2, :) = Xi(1) * (C(1) - C(1)*exp(-(C(1) + C(2))*time)) / (C(1) + C(2));
% ODE-45 Solution
[rktime, rkconc] = ode45(@(t, y) revrxn(t, y, C), time, Xi);
% Errors
% Error w.r.t. Direct Method and Closed Form Solution
cferror = X - cfs(:, end - 1);
% Error w.r.t. Direct Method and ODE-45 Solution
rkerror = X' - rkconc(1:end - 1, :);
% Error w.r.t. Closed Form Solution and ODE-45 Solution
cfrkerror = cfs - rkconc';

%% Plot the Direct Method Solution
figure;
plot(time(:, 1:end - 1), X(1, :), "Color", "green");
hold on
plot(time(:, 1:end - 1), X(2, :), "Color", "blue");
legend(["A", "A*"]);
xlabel("Time");
ylabel("Concentration");
title("Direct Method");

%% Plot Analyzing Differences Between Direct Method, Closed Form and ODE-45 Solution
figure;
subplot(2, 3, 1)
plotconc(time(:, 1:end - 1), X, colors(1:2), labels(1:2), "Direct Method Solution");
subplot(2, 3, 2)
plotconc(rktime, rkconc', colors(1:2), labels(1:2), "ODE45 Solution");
subplot(2, 3, 3)
plotconc(time, cfs, colors(1:2), labels(1:2), "Closed Form Solution");
subplot(2, 3, 4)
plotconc(time(:, 1:end - 1), cferror, colors(3:4), labels([1, 3]), strcat("Sum of Errors(CFS): ", num2str(sum(abs(cferror(1, :))))));
subplot(2, 3, 5)
plotconc(rktime(1:end - 1), rkerror', colors(3:4), labels([1, 3]), strcat("Sum of Errors(ODE45): ", num2str(sum(abs(rkerror(:, 1))))));
subplot(2, 3, 6)
plotconc(rktime, cfrkerror, colors(3:4), labels([1, 3]), strcat("Sum of Errors(ODE45): ", num2str(sum(abs(cfrkerror(:, 1))))));

%% Plot Analyzing Tau-Value With Respect to Time
figure;
plot(1:length(tau_arr), tau_arr, Color = "blue");
hold on;
plot(1:length(tau_arr), movmean(tau_arr, 10), Color = "green");
xlabel("Iteration #");
ylabel("Tau (a.u.)");
title(strcat("Tau Value, Average: ", num2str(mean(tau_arr))));
xlim([0 length(tau_arr) + 100]);
ylim([-0.5 1.5]);

%% Function for First Order Reversible Reaction
function Xv = reaction(Xv, mu)
  switch mu
    case 1
      Xv(1) = Xv(1) - 1;
      Xv(2) = Xv(2) + 1;
    case 2
      Xv(1) = Xv(1) + 1;
      Xv(2) = Xv(2) - 1;
  end
end

%% Function for Describing the ODE Behavior to the ODE-45 Solver
function dydt = revrxn(t, y, C)
  dA1dt = C(2) * y(2) - C(1) * y(1);
  dA2dt = C(1) * y(1) - C(2) * y(2);
  dydt = [dA1dt; dA2dt];
end

%% Function to Update the C.H Matrix
function [H, Ch] = update_ch(C, X)
  H = X;
  Ch = C.*H';
end

%% Plotting Function to Plot Concentrations
function plotconc(x, y, colors, labels, stitle)
  for i = 1:2
    plot(x, y(i, :), Color = colors(i));
    hold on;
  end
  xlabel(labels(1));
  ylabel(labels(2));
  title(stitle);
  legend(["[A]", "[A*]"]);
  hold off;
end