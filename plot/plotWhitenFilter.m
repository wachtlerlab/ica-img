function [ ] = plotWhitenFilter(filter, doprint)

if ~exist('doprint', 'var') || isempty(doprint); doprint = 0; end;

pos = [400, 400];
fsize = [300, 300];

data = filter.W(25,:);
data = processData(data);
fhS1 = plotImg(data, pos, fsize, 'L');
fhS2 = plotSlice(data, pos, fsize, 'L');

data = filter.W(25+49,:);
data = processData(data);
fhM1 = plotImg(data, pos, fsize, 'M');
fhM2 = plotSlice(data, pos, fsize, 'M');


data = filter.W(25+2*49,:);
data = processData(data);
fhL1 = plotImg(data, pos, fsize, 'S');
fhL2 = plotSlice(data, pos, fsize, 'S');

if doprint
    figname = 'whiten-filter-';
    
    printFig(fhS1, [figname 'L']);
    printFig(fhS2, [figname 'Lcut']);
    
    printFig(fhM1, [figname 'M']);
    printFig(fhM2, [figname 'Mcut']);
    
    printFig(fhL1, [figname 'S']);
    printFig(fhL2, [figname 'Scut']);
end

end

function printFig(fh, figname)
fsize = 4.5;

set(fh, 'PaperUnits', 'centimeters', 'PaperSize', [fsize fsize], 'PaperPosition', [0, 0, fsize, fsize])
print(fh, '-depsc2', '-r300', '-loose', figname)

end

function [X] = processData(data)
data = data/max(abs(data(:)));
X = reshape(data, 7, 7, 3);
end

function [fh] = plotSlice(data, pos, fsize, title)
fh = figure('Name', title, 'Color', 'w', 'Position', horzcat(pos, fsize));
hold on;
plot(data(4,:,1), 'r')
plot(data(4,:,2), 'g')
plot(data(4,:,3), 'b')

xlim([1 7])
ylim([-1 1])

end

function [fh] = plotImg(data, pos, fsize, title)

fh = figure('Name', title, 'Color', 'w', 'Position', horzcat(pos, fsize));
data = 0.5+0.5*data;
X = reshape(data, 7, 7, 3);
imagesc(X)
axis equal tight image off
set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])

end