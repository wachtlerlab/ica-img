function [ fh ] = plotAL2vsKurt(A)

pos = [300, 300];
fs = [500, 500];
figtitle = ['Fit ' num2str(bfnr) ' ' num2str(imgnr)];
curfig = figure('Name', figtitle, 'Position', horzcat(pos, fs), ...
    'Color', 'w', 'PaperType', 'A4');
set(0,'CurrentFigure', curfig);

scatter(A.norm, A.beta, 20, 'k', 'filled');


end
