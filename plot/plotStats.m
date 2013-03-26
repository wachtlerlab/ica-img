function [ Result ] = plotStats(Model)

%addpath('functions/home/tewon/Matlab/Lewicki/expwrcode/stats')
%addpath('functions/home/lewicki/matlab/oc')
ds = Model.ds;
Result.iter = (ds.npats / ds.blocksize) * (ds.nclusters-2) + 1;
Result = samplePats(Result, ds)
Result.S = pinv(Model.A)*Result.D;

Model.prior.beta = Model.beta;
Model.prior.sigma = ones(size(Model.beta, 1));
Model.prior.mu = zeros(size(Model.beta, 1));

figure(2);
figure(3);

Result.iter = 1;
updateDisplay (Model, Result, 1);


end

