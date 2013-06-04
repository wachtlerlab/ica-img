function logL = calcLPXA(X,S,Model)

N = size(X,2);		% number of patterns
A = Model.A;
[~,M] = size(A);

% for a single pattern x, 
%    log P(x|A) =  log P(S) - log |det A|

logPS = 0;
for m=1:M
  logPS = logPS + sum(expwrpdfln(S(m,:),0,1,Model.beta(m)));
end
logL = logPS - N*log(abs(det(A)));
end