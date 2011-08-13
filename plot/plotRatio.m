function [ hf ] = plotRatio (Model, range, figHandle)

if (nargin < 3)
  hf = figure ('Name', ['Basis Functions: ', Model.name], ...
   'Position', [0, 0, 800, 1000], 'Color', 'w', 'PaperType', 'A4');
else
  set (0, 'CurrentFigure', figHandle);
  hf = figHandle;
end

Model = sortModelA (Model);
A = Model.A;
A = normABf (A);

[~,M] = size(A);

patchSize = Model.patchSize;
dataDim = Model.dataDim;

if nargin < 2 || isempty (range)
  range = 1:M;
end

for idx = range
    bf = A (:, idx);
    R = reshape (bf, dataDim, patchSize * patchSize);
    for n = 1:dataDim
        v = max (R(n,:)) - min (R(n,:));
        l(idx,n) = v;
    end

end

len = length(l);
midpoint = ceil ((len)/2);

colormap ('gray')
subplot(1,2,1);
range = 1:midpoint;
imagesc (1:4,range,l(range,:))

subplot(1,2,2);
range = midpoint+1:len;
imagesc (1:4,range, l(range,:))

end
