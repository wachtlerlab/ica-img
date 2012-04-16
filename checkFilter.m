function checkFilter (modelId)

if nargin < 1
modelId = 'color_cs_rect';
end

fprintf ('Checking filter for [%s]\n', modelId);

clear Model fitPar dispPar Result;

[ Model, FitParam, DisplayParam, DataParam ] = loadConfig (modelId);

fprintf ('Config Id: %s\n', Model.cfgId(1:7));

images = prepareImages (DataParam);
displayImages (images, DataParam);

end