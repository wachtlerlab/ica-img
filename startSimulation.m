if isempty (modelId)
  modelId = 'color_plain';
end

[Model, Result] = fitModel (modelId);

