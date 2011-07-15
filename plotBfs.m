function [hf] = plotBfs (Model, figHandle)

dataDim = Model.dataDim;

if (nargin < 2)
  figHandle = figure ('Name', ['Basis Functions: ', Model.name], ...
    'Position', [0, 0, 800, 1000], 'Color', 'w', 'PaperType', 'A4');
end

if dataDim == 3
  hf = plotABfLMS (Model, figHandle);
else
  hf = plotABfCSRect (Model, 1, 15, figHandle);
end

end

