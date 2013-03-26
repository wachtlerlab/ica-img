function [ curfig ] = plotACreateFig(A, name, landscape, fs)

if ~exist('fs', 'var') || isempty(fs); fs = [900, 600]; end;
if ~exist('landscape', 'var') || isempty(fs); landscape = 0; end;

if landscape
    fs = sort(fs);
end
pos = [300, 300];

figtitle = [name ' ' A.name];
curfig = figure('Name', figtitle, 'Position', horzcat(pos, fs), ...
    'Color', 'w', 'PaperType', 'A4');
set(0,'CurrentFigure', curfig);


end

