clc; clear; close all;
addpath(genpath('./src'));

%%
t0 = 0;
tend = 20;
dt = 0.1;
ca0 = 1;
kf = 1;
kr = 0.5;
time = t0:dt:tend;
afwd = zeros(6, size(time, 2));
arev = zeros(6, size(time, 2));
expconc = zeros(8, size(time, 2));

%%
expconc(1, :) = ca0 * (kr + kf*exp(-(kf + kr)*time)) / (kr + kf);
expconc(2, :) = ca0 * (kf - kf*exp(-(kf + kr)*time)) / (kr + kf);
expconc(3:4, :) = rk1([kf, kr], t0, tend, dt, [ca0, 0], 1);

%%
figure;
subplot(1, 3, 1)
colors = ["b", "g"];
labels = ["Time", "Concentration", "Error"];
for i = 1:2
  plot(time, expconc(i, :), Color = colors(i));
  hold on;
end
xlabel(labels(1));
ylabel(labels(2));
title("Analytical Solution");