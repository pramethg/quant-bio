function cells = rk1(alpha, t0, tend, dt, y0, model)
  niter = (tend - t0) / dt;
  switch model
    case 0
      cells = zeros(1, niter + 1);
      cells(1) = y0;
      for i = 1:niter
        cells(i + 1) = cells(i) + (dt * alpha * cells(i));
      end
    case 1
      cells = zeros(2, niter + 1);
      cells(1, 1) = y0(1);
      cells(2, 1) = y0(2);
      for i = 1:niter
        dCa = dt * (alpha(2) * cells(2, i) - alpha(1) * cells(1, i));
        cells(1, i + 1) = cells(1, i) + dCa;
        cells(2, i + 1) = cells(2, i) - dCa;
      end
  end
end