function [dA, A] = calcDeltaA(S,Model, onGPU)

%tM = tic;

if onGPU
  [dA, A] = calcOnGPU (S, Model);
else
  [dA, A] = calcOnHost (S, Model);
end

%te = toc (tM);
%fprintf ('dA done in %f [gpu: %d]\n', te, onGPU);

end

function [dA, A] = calcOnGPU(S, Model)

A = Model.A;
[~,M] = size(A);
npats = size(S,2);
mp = Model.prior;
Z = zeros (size(S));

for m=1:M
  s = S(m,:) - mp.mu(m);
  q = (2/(1+mp.beta(m)));
  c = (gamma(3/q)/gamma(1/q)).^(q/2);
  Z(m,:) = -(q*c/(mp.sigma(m).^q)) * abs(s).^(q-1) .* sign(s);
end

gA = gpuArray (A);
gZ = gpuArray (Z);
gS = gpuArray (S);
gnpats = gpuArray (npats);

dA = -gA*gZ*gS' - gnpats*gA;
dA = dA/gnpats;

A = gA;

end

function [dA, A] = calcOnHost (S, Model)

A = Model.A;
[L,M] = size(A);
npats = size(S,2);
mp = Model.prior;

dA = zeros(size(A));
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
