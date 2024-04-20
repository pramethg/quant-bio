function totalLogLik = logLikelihood(allData, parameters)
  sigmaT = 4.5;
  sigmaI = 0.5;
  sigmalogV = 0.45;
  time = allData(:, 1)'; 
  Y0 = allData(1, 2:end);
  [tModel, YModel] = ode45(@(t, Y) modelCOVID(t, Y, parameters), time, Y0);
  TModel = interp1(tModel, YModel(:, 1), time);
  IModel = interp1(tModel, YModel(:, 2), time);
  VModel = interp1(tModel, YModel(:, 3), time);
  resT = allData(:, 2) - TModel;
  resI = allData(:, 3) - IModel;
  resV = allData(:, 4) - VModel;
  logLikT = -0.5 * sum((resT / sigmaT).^2);
  logLikI = -0.5 * sum((resI / sigmaI).^2);
  logLikV = -0.5 * sum((resV / sigmalogV).^2);
  totalLogLik = logLikT + logLikI + logLikV;
end