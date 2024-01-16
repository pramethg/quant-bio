function cells = rk2(alpha, t0, tend, dt, y0, model)
  niter = (tend - t0) / dt;
  switch model
    case 0
      cells = zeros(1, niter + 1);
      cells(1) = y0;
      for i = 1:niter
        k1 = dt*alpha*cells(i);
        k2 = dt*alpha*(cells(i)+(dt / 2)*k1);
        cells(i + 1) = cells(i) + (k1 + k2) / 2;
      end
    case 1
      cells = zeros(2, niter + 1);
      cells(1, 1) = y0(1);
      cells(2, 1) = y0(2);
      for i = 1:niter
        k1 = dt * (alpha(2) * cells(2, i) - alpha(1) * cells(1, i));
        k2 = dt * (alpha(2) * (cells(2, i) + (dt / 2) * k1) - alpha(1) * (cells(1, i) + (dt / 2) * k1));
        cells(1, i + 1) = cells(1, i) + (k1 + k2) / 2;
        cells(2, i + 1) = cells(2, i) - (k1 + k2) / 2;
      end
  end
end