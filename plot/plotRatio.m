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

ha = tight_subplot(1, 2, .1, 0.1);

midpoint = ceil ((length(l)-1)/2)

set (gcf, 'CurrentAxes', ha(1));
colormap ('gray');
hold on;
imagesc (l(1:midpoint,:));
ylim ([min(range) range(midpoint)]);
xlim ([1 4]);
set(gca,'YTickLabel', min(range):range(midpoint))
txt = sprintf ('%d:%d', min(range), range(midpoint));
title (ha(1), txt);


set (gcf, 'CurrentAxes', ha(2));
colormap ('gray');
hold on;
imagesc (l(midpoint+1:end,:));
%ylim ([range(midpoint+1) max(range)]);
ylim ([min(range) range(midpoint)]);
xlim ([1 4]);
set(gca,'YTickLabel', range(midpoint+1):max(range))
txt = sprintf ('%d:%d', range(midpoint+1), max(range));
title (ha(2), txt);


end
