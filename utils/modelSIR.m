function dYdt = modelSIR(t, Y, params)
  S = Y(1);
  R = Y(2);
  I = Y(3);
  b = params(1);
  k = params(2);
  dSdt = b * S * I;
  dRdt = k * I;
  dIdt = (b * S * I) - (k * I);
  dYdt = [dSdt; dRdt; dIdt];
end