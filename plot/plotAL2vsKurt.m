function [ fig ] = plotAL2vsKurt(A)

pos = [300, 300];
fs = [500, 500];
fig = figure('Name', 'L2 versus kurtosis', 'Position', horzcat(pos, fs), ...
    'Color', 'w', 'PaperType', 'A4');
set(0,'CurrentFigure', fig);

scatter(A.norm, A.kurt, 20, 'k', 'filled');
xlabel('L^2 norm')
ylabel('kurtosis')

end
