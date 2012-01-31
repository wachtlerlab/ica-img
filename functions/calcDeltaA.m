function [dA, A] = calcDeltaA(S, Model)

A = Model.A;
[~,M] = size(A);
npats = size(S,2);
mp = Model.prior;

Z  = zeros(size(S));

for m=1:M
  s = S(m,:) - mp.mu(m);
  q = 2/(1+mp.beta(m));
  c = (gamma(3/q)/gamma(1/q))^(q/2);
  Z(m,:) = -(q*c/(mp.sigma(m)^q)) * abs(s).^(q-1) .* sign(s);
end

dA = -A*Z*S' - npats*A;

% normalize by the number of patterns
dA = dA/npats;

end
