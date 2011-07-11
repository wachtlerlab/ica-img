function [ Model, Results ] = fitModel (modelId)

clear Model fitPar dispPar Result;

tStart = tic;

addpath (genpath ('functions'));
figure(1)
figure(2)
figure(3)

[ Model, FitParam, DisplayParam, DataParam ] = loadConfig (modelId);

if FitParam.startIter <= 1
  Model.A = eye(size(Model.A));		% again, we assume A is square
end

% infer matrix
%
[Model, Results] = fitModel_color_plain(Model, FitParam, DisplayParam, DataParam);

% time reporting
telapsed = toc (tStart);
fprintf (['Total time: (',num2str(telapsed),')\n']);

end

