clc; clear; close all;
addpath(genpath('./src'));

%%
alpha = 0.02;
t0 = 0;
tend = 200;
dt = 1;
cells0 = 2;
time = t0:dt:tend;
cellarr = zeros(6, size(time, 2));

%%
expcells = cells0 * exp(alpha * (time));
cellarr(1, :) = rk1(alpha, t0, tend, dt, cells0, 0);
cellarr(3, :) = rk2(alpha, t0, tend, dt, cells0, 0);
cellarr(5, :) = rk4(alpha, t0, tend, dt, cells0, 0);
titles = ["Euler's Method", "Euler's Error", "RK-2 Method", "RK-2 Error", "RK-4 Method", "RK-4 Error"];
for idx = 2:2:6
  cellarr(idx, :) = expcells - cellarr(idx - 1, :);
end

%%
figure;
for i = 1:6
  subplot(3, 2, i)
  plot(time, cellarr(i, :));
  xlabel("Time (minutes)");
  if mod(i, 2) == 0
    ylabel("Error (a.u.)");
    title(strcat(titles(i), " Sum: ", num2str(sum(cellarr(i, :)))));
  else
    ylabel("Cells (a.u.)");
    title(titles(i));
  end
end