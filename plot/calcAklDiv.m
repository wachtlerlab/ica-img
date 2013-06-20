function [ kest, ui ] = calcAklDiv(A, plotData)

if ~exist('plotData', 'var') || isempty(plotData); plotData = 0; end;

L = 20;  % was 49

pcs = A.pcs;
tt = atan2(pcs(2,:), pcs(1,:));
X = mod(tt + pi, pi);
c = 0:pi/(L-1):pi;
n = histc(X, c);

Xn = n/sum(n) + eps; % + eps is there to avoid log2(0) -> -Inf 
Xu = ones(1, L)/L;

kest = kldiv(c, Xu, Xn);

if plotData
    figure;
    hold on;
    color = [1 0 0];
    bar(c, Xu, 'EdgeColor', color, 'FaceColor', color);
    
    color = [0 0 0];
    bar(c, Xn, 'EdgeColor', color, 'FaceColor', color);
    
    xlim([0 pi])
end

gain = 0.5;

ui =  2 - (2./(1+(exp(-0.1*kest))));

end
