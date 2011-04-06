clear Model fitPar dispPar Result;

tic;

addpath (genpath ('functions'));
figure(1)
figure(2)
figure(3)

Model.name   = 'color_plain_cs';  % 7x7 LMS Filtered 

% Model parameters
%
L            = 7*7*3;  % 3 chromatic dimensions
M            = L;		% expwr only allows square A for now
Model.A      = zeros(L,M);

% params for exponential power prior
ExPwr.mu     = zeros(M,1);
ExPwr.sigma  = ones(M,1);
ExPwr.beta   = 0.5*ones(M,1);	% assume slightly supergaussian (tanh)
ExPwr.a      = 2;
ExPwr.b      = 2;
ExPwr.tol    = 0.1;

Model.prior  = ExPwr;

% parameters related to fitting
%
FitParam.startIter      = 1;
FitParam.blocksize      = 100;
FitParam.priorAdaptSize = 5000;	% how many coefs to collect before adapting
FitParam.npats          = 40000;	% number of pats in new dataset
FitParam.dataFn         = 'getImageData';  % function that generates dataset
FitParam.dataFnArgs     = [sqrt(L), FitParam.npats];
FitParam.iterPts        =   [  1,    1000,   5000,  10000 30000];
FitParam.epsilon        = 2*[ 0.02,  0.01,  0.005,  0.001 0.0005];
FitParam.saveFreq       = 100;

FitParam.maxIters = max(FitParam.iterPts);

% params related to displaying of results and fitting progress
%
DisplayParam.plotflag       = 1;
DisplayParam.updateFreq     = 50;
DisplayParam.maxPlotVecs    = M;

% params related to generating the datasets, e.g. filtering
DataParam.doFilter = true;
DataParam.filter = mexican_hat (3, 20, 1, 6);

if FitParam.startIter <= 1
  Model.A = eye(size(Model.A));		% again, we assume A is square
end

% infer matrix
%
[Model, Results] = fitModel_color_plain(Model, FitParam, DisplayParam, DataParam);

% time reporting
telapsed=toc;
fprintf (['Total time: (',num2str(telapsed),')\n']);
