clc; clear; close all;
time = [];
tau_arr = [];
tstop = 100;
tspan = [0 tstop];
time(1) = tspan(1);
Xi = [100 0];
X = zeros(length(Xi), tstop);
X(:, 1) = Xi;
C = [0.1 0.1];
H = update_ch(C, Xi);
Ch = C.*H';
i = 2;
t = 0;
colors = ["b", "g", "yellow", "cyan"];
labels = ["Time", "Concentration", "Error"];

%%
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

%%
cfs(1, :) = Xi(1) * (C(2) + C(1)*exp(-(C(1) + C(2))*time)) / (C(1) + C(2));
cfs(2, :) = Xi(1) * (C(1) - C(1)*exp(-(C(1) + C(2))*time)) / (C(1) + C(2));
[rktime, rkconc] = ode45(@(t, y) revrxn(t, y, C), time, Xi);
cferror = X - cfs(:, end - 1);
rkerror = X' - rkconc(1:end - 1, :);

%%
figure;
plot(1:length(time), time);

%%
figure;
plot(1:length(tau_arr), tau_arr, Color = "blue");
hold on;
plot(1:length(tau_arr), movmean(tau_arr, 10), Color = "green");
xlabel("Iteration #");
ylabel("Tau (a.u.)");
title(strcat("Tau Value, Average: ", num2str(mean(tau_arr))));
xlim([0 length(tau_arr) + 100]);
ylim([-0.5 1.5]);

%%
figure;
subplot(2, 3, 1)
plotconc(time(:, 1:end - 1), X, colors(1:2), labels(1:2), "Direct Method Solution");
subplot(2, 3, 2)
plotconc(rktime, rkconc', colors(1:2), labels(1:2), "ODE45 Solution");
subplot(2, 3, 3)
plotconc(time, cfs, colors(1:2), labels(1:2), "Closed Form Solution");
subplot(2, 3, 4)
plotconc(time(:, 1:end - 1), cferror, colors(3:4), labels([1, 3]), strcat("Error(CFS): ", num2str(sum(abs(cferror(1, :))))));
subplot(2, 3, 5)
plotconc(rktime(1:end - 1), rkerror', colors(3:4), labels([1, 3]), strcat("Error(ODE45): ", num2str(sum(abs(rkerror(:, 1))))));

%%
figure;
plot(time(:, 1:end - 1), X(1, :), "Color", "green");
hold on
plot(time(:, 1:end - 1), X(2, :), "Color", "blue");
legend(["A", "A*"]);
xlabel("Time");
ylabel("Concentration");
title("Direct Method");

%%
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
%%
function dydt = revrxn(t, y, C)
  dA1dt = C(2) * y(2) - C(1) * y(1);
  dA2dt = C(1) * y(1) - C(2) * y(2);
  dydt = [dA1dt; dA2dt];
end


%% 
function [H, Ch] = update_ch(C, X)
  H = X;
  Ch = C.*H';
end

%%
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