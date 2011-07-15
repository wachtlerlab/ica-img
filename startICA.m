function [Model] = startICA (modelId, autosave)

if nargin < 1
  modelId = 'color_cs_rect_1';
end;

fprintf ('Starting simulation for %s', modelId)
[Model, Result] = fitModel (modelId);

Model.Result = Result;

if autosave
  saveResults (Model);
  fprintf ('Saved results to %s\n', filename);
end

end
