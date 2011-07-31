function dA = calcDeltaA(S,Model)

%
%
%
%

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

%tM = tic;
gA = gpuArray (A);
gZ = gpuArray (Z);
gS = gpuArray (S);
gnpats = gpuArray (npats);

gdA = -gA*gZ*gS' - gnpats*gA;
gdA = gdA/gnpats;
dA = gather (gdA);

%te = toc (tM);
%printf ('dA done in %f (CPU)\n', te);


end