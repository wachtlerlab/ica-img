function [ fig ] = plotACoeffStats(A, Act)

N = 25;

[nrows, ncols] = plotCreateGrid(N);
landscape = 0;

fig = plotACreateFig(A, 'Coeff Stats', landscape, [1200, 800]);
hb = tight_subplot(nrows, ncols, [.01 .01], [.01 .01]);

for idx=1:N
    set (gcf, 'CurrentAxes', hb(idx));
    hold on
    
    %hist(Act.w(:, idx), 40);
    mu = 0;
    sigma = 1;
    beta = A.beta(idx);
    
    fprintf('ExPwr: mu=%5.2f  sigma=%5.2f  beta=%+5.2f\n', mu,sigma,beta);
    
    % equivalent params for gexp
    p = 2/(1 + beta);
    c = (gamma(3/p) / gamma(1/p))^(p/2);
    s = sigma * c^(-1/p);
    
    fprintf('GExp: s=%5.2f  p=%5.2f\n',s,p);
    
    %n = 1000;
    %xrange = 5*sigma;
    %dx = 2*xrange/n;
    %x = -xrange:dx:xrange;
    %y = expwrpdf(x,mu,sigma,beta);
    [y, x] = hist(Act.w(:, idx),100);
    plot(x,y, 'k');
    axis tight;
    axis off;
    %title(sprintf('ExPwr(y | \\mu=%g, \\sigma=%g, \\beta=%g)',mu,sigma,beta));
    k = expwrkur(beta);
    fprintf('k=%5.3f\n',k);
    text(0, max(y)*1.1, num2str(k))
    ylim([0, max(y)*1.2])
  
end

end

function [k] = calcKurtosis(beta)

pb = 1 + beta;
fhb = 5/2*pb;

k = (gamma(fhb)*gamma(0.5*pb) / gamma(fhb)^2) - 3;

end
