clc; clear; close all;
addpath(genpath('./src'));

%%
t0 = 0;
tend = 10;
dt = 0.001;
ca0 = 1;
kf = 1;
kr = 0.5;
time = t0:dt:tend;
afwd = zeros(6, size(time, 2));
arev = zeros(6, size(time, 2));
expconc = zeros(8, size(time, 2));
errors = NaN(6, size(time, 2));

%%
expconc(1, :) = ca0 * (kr + kf*exp(-(kf + kr)*time)) / (kr + kf);
expconc(2, :) = ca0 * (kf - kf*exp(-(kf + kr)*time)) / (kr + kf);
expconc(3:4, :) = rk1([kf, kr], t0, tend, dt, [ca0, 0], 1);
expconc(5:6, :) = rk2([kf, kr], t0, tend, dt, [ca0, 0], 1);
expconc(7:8, :) = rk4([kf, kr], t0, tend, dt, [ca0, 0], 1);

for i = 0:2
  errors(2*i+1:2*i+2, :) = abs(expconc(1:2, :) - expconc(2*i+3:2*i+4, :));
end

%%
colors = ["b", "g"];
labels = ["Time", "Concentration", "Error"];

figure;
subplot(2, 4, 1)
plotconc(time, expconc(1:2, :), labels(1:2), colors, "Analytical Solution");
subplot(2, 4, 2)
plotconc(time, expconc(3:4, :), labels(1:2), colors, "Euler's Method Solution");
subplot(2, 4, 3)
plotconc(time, expconc(5:6, :), labels(1:2), colors, "Runge Kutta 2nd Order Solution");
subplot(2, 4, 4)
plotconc(time, expconc(7:8, :), labels(1:2), colors, "Runge Kutta 4th Order Solution");
ploterrror(time, errors, ["Euler's", "RK-2", "RK-4"], labels([1, 3]));

%%
function plotconc(x, y, labels, colors, stitle)
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

%%
function ploterrror(x, y, stitles, labels)
  for i = 0:2
    subplot(2, 4, 5 + i)
    plot(x, y(2*i+1, :), Color = "red");
    hold on;
    plot(x, y(2*i+2, :), Color = "cyan");
    title(strcat("Error ", stitles(i + 1), " [", num2str(round(sum(y(2*i+1, :)), 3)), ", ", num2str(round(sum(y(2*i+2, :)), 3)), "]"));
    xlabel(labels(1));
    ylabel(labels(2));
    legend(["[A]", "[A*]"])
  end
end