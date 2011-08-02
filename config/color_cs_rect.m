%
Mode.comment = 'On/Off Center-Surround with simple Filter (center: 1.1)';

% params related to generating the datasets, e.g. filtering
%
simpleFilter = ones (3, 3) * 0.1250;
simpleFilter(2,2) = 1.1;

DataParam.fileList  = [ 'ashton3 '; 'fuschia '; 'moss    '; 'plaza   '; ...
                        'red1    '; 'rwood   '; 'valley  '; 'yellow1 ' ];
DataParam.dataDir   = fullfile ('..', 'data', 'bristol');
DataParam.patchSize = 7;
DataParam.dataDim   = 6; %this is post filtering
DataParam.doFilter  = true;
DataParam.filter    = simpleFilter;
DataParam.filterFn  = 'filterCSRecConv';
DataParam.doDebug   = true;
DataParam.doLog     = true;

% Model parameters
%
L               = (DataParam.patchSize^2) * DataParam.dataDim;
M               = L;		% expwr only allows square A for now
Model.A         = randn (L, M) + 0.5 * eye(L, M); %zeros(L,M);
Model.patchSize = DataParam.patchSize;
Model.dataDim   = DataParam.dataDim;

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
FitParam.iterPts        =     [  1,    1000,  5000, 10000, 30000,  60000 ];
FitParam.epsilon        = 20 *[ 0.02,  0.01, 0.005, 0.001, 0.0005, 0.0001];
FitParam.saveflag       = 1;
FitParam.saveFreq       = 100;

FitParam.maxIters = max(FitParam.iterPts);

% params related to displaying of results and fitting progress
%
DisplayParam.plotflag       = 1;
DisplayParam.updateFreq     = 50;
DisplayParam.maxPlotVecs    = M;