function [table] = expwrchkbeta()

N = 10e5;
M = 10;

x = -M:1/N:M;

betas = 0:0.5:6;
table = zeros(length(betas), 4);

for idx = 1:length(betas);
    beta = betas(idx);
    fprintf('beta: %2.1f\n', beta)
    y = expwrpdf(x, 0, 1, beta);
    k_a = expwrkur(beta);
    k_b = stdkurt(y);
    k_c = kurtosis(y);
   
   table(idx, :) = [beta, k_a, k_b, k_c];
end

end