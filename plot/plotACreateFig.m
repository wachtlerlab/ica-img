function [ curfig ] = plotACreateFig(A, name, landscape, fs)

if ~exist('fs', 'var') || isempty(fs); fs = [900, 600]; end;
if ~exist('landscape', 'var') || isempty(fs); landscape = 0; end;

if landscape
    fs = sort(fs);
end
pos = [300, 300];

if isstruct(A)
    figtitle = [name ' ' A.name];
else
    figtitle = A;
end

curfig = figure('Name', figtitle, 'Position', horzcat(pos, fs), ...
    'Color', 'w', 'PaperType', 'A4');
set(0,'CurrentFigure', curfig);


end

