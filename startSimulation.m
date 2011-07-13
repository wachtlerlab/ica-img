if exist ('modelId', 'var') == 0
  modelId = 'color_cs_rect_1';
end

fprintf ('Starting simulation for %s', modelId)
[Model, Result] = fitModel (modelId);

