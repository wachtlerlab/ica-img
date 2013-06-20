function [ kest, ui ] = calcKLDiv(tt)
if ~exist('plotData', 'var') || isempty(plotData); plotData = 0; end;

L = 49;  % was 49

X = mod(tt + pi, pi);
c = 0:pi/(L-1):pi;
n = histc(X, c);

Xn = n/sum(n) + eps; % + eps is there to avoid log2(0) -> -Inf 
Xu = ones(1, L)/L;

kest = kldiv(c, Xu, Xn);
ui =  2 - (2./(1+(exp(-0.1*kest))));

end

