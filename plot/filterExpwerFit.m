function [ epp, wfilter ] = filterExpwerFit(wi, border)

if ~exist('border', 'var'); border = 0.2; end;

mu    = mean(wi);
sigma = std(wi);

beta = expwrmapbeta (wi, mu, sigma, 2, 2, 0.1);
epp = [mu, sigma, beta];

if (abs(mu) > 0.01) || beta < 0
    wfilter = ones(size(wi));
    return;
end

stwide = 0.05;
intwidth = 15; % was 7 (27.5) or 15
 
x = (mu-intwidth):stwide:(mu+intwidth);

intx = @(b) integral(@(x) expwrpdf(x, mu, sigma, beta), -Inf, b);
yf = arrayfun(intx, x);

idx_a = find(yf > (border*0.5), 1);
idx_b = find(yf > 1 - (border*0.5), 1);

lowerb = x(idx_a);
upperb = x(idx_b);

if isempty(lowerb) || isempty(upperb)
    wfilter = zeros(size(wi));
    return;
end

wfilter = wi < lowerb | wi > upperb;

% if nargout > 2
%     bounds = zeros(2,2);
%     bounds(1,:) = [yf(idx_a), yf(idx_b)];
%     bounds(2,:) = [lowerb, upperb];
% end


end

