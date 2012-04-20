function [ prior ] = createPrior (cfg, dim)

beta = cfg.beta;
sigma = cfg.sigma;
mu = cfg.mu;

prior.mu        = mu * ones(dim,1);
prior.sigma     = sigma * ones(dim,1);
prior.beta      = beta * ones(dim,1);	% assume slightly supergaussian (tanh)
prior.a         = cfg.a;
prior.b         = cfg.b;
prior.tol       = cfg.tol;
prior.adaptSize = cfg.adapt_size;

end
