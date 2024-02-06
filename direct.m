clc; clear; close all;

time = [];
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

%%
while (t < tstop)
  r1 = rand();
  r2 = rand();
  A = sum(Ch(:));
  tau = 1 / (A + 1e-5) * log(1 / (r1 + 1e-5));
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
function [H, Ch] = update_ch(C, X)
  H = X;
  Ch = C.*H';
end