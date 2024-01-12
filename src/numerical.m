%%
alpha = 0.02;
t0 = 0;
tend = 200;
dt = 1;
cells0 = 2;

time = t0:dt:tend;
cells = zeros(6, size(time, 2));
expcells = cells0 * exp(alpha * (time));
cells(1, :) = euler(alpha, t0, tend, dt, cells0);
cells(3, :) = rk2(alpha, t0, tend, dt, cells0);
cells(5, :) = rk4(alpha, t0, tend, dt, cells0);
titles = ["Euler's Method", "Euler's Error", "RK-2 Method", "RK-2 Error", "RK-4 Method", "RK-4 Error"];
for idx = 2:2:6
    cells(idx, :) = expcells - cells(idx - 1, :);
end

figure;
for i = 1:6
    subplot(3, 2, i)
    plot(time, cells(i, :));
    xlabel("Time (minutes)");
    if mod(i, 2) == 0
        ylabel("Error (a.u.)");
        title(strcat(titles(i), " Sum: ", num2str(sum(cells(i, :)))));
    else
        ylabel("Cells (a.u.)");
        title(titles(i));
    end
end

%%

function cells = euler(alpha, t0, tend, dt, y0)
    niter = (tend - t0) / dt;
    cells = zeros(1, niter + 1);
    cells(1) = y0;
    for i = 1:niter
        cells(i + 1) = cells(i) + (dt * alpha * cells(i));
    end
end

%% 

function cells = rk2(alpha, t0, tend, dt, y0)
    niter = (tend - t0) / dt;
    cells = zeros(1, niter + 1);
    cells(1) = y0;
    for i = 1:niter
        k1 = dt*alpha*cells(i);
        k2 = dt*alpha*(cells(i)+(dt / 2)*k1);
        cells(i + 1) = cells(i) + (k1 + k2) / 2;
    end
end

%% 

function cells = rk4(alpha, t0, tend, dt, y0)
    niter = (tend - t0) / dt;
    cells = zeros(1, niter + 1);
    cells(1) = y0;
    for i = 1:niter
        k1 = dt*alpha*cells(i);
        k2 = dt*alpha*(cells(i)+(dt / 2)*k1);
        k3 = dt*alpha*(cells(i)+(dt / 2)*k2);
        k4 = dt*alpha*(cells(i)+dt*k3);
        cells(i + 1) = cells(i) + (k1 + 2*k2 + 2*k3 + k4) / 6;
    end
end
