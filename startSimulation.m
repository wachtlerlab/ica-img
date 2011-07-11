if exist ('modelId', 'var') == 0
  modelId = 'color_plain';
end

[Model, Result] = fitModel (modelId);

