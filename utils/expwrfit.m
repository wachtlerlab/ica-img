function [mu, sigma, beta] = expwrfit( data )

mu    = mean(data);
sigma = std(data);

beta = expwrmapbeta (data, mu, sigma, 2, 2, 0.1);
end

