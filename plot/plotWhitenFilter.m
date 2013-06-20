function [ ] = plotWhitenFilter(filter, doprint)

if ~exist('doprint', 'var') || isempty(doprint); doprint = 0; end;

patchsize = filter.patchsize;
sn = size(filter.W);

if filter.perchan
    nchan = 1;
else
    nchan = 3;
end

pos = [400, 400];
fsize = [300, 300];

if filter.perchan
    chanid = ['S', 'M', 'L'];
else
    chanid = ['L', 'M', 'S'];
end

figs = zeros(nchan, 2);

figname = 'whiten-filter-';
if ischar(doprint)
    figname = [doprint figname];
end
        
for ch = 1:3

    if filter.perchan
        data = filter.W(25, :, ch);
        data = data/max(filter.W(25, :));
    else
        data = filter.W(25+(ch-1)*49,:);
        data = data/max(abs(data(:)));
    end
    
    
    data = reshape(data, 7, 7, nchan);
    chid = chanid(ch);
    figs(ch, 1) = plotImg(data, pos, fsize, chid, nchan);
    figs(ch, 2) = plotSlice(data, pos, fsize, chid, nchan);
    if doprint

        printFig(figs(ch, 1), [figname chid]);
        printFig(figs(ch, 2), [figname chid 'cut']);
    end
end


% data = filter.W(25,:);
% data = processData(data, nchan);
% fhS1 = plotImg(data, pos, fsize, 'L');
% fhS2 = plotSlice(data, pos, fsize, 'L', nchan);
% 
% data = filter.W(25+49,:);
% data = processData(data, nchan);
% fhM1 = plotImg(data, pos, fsize, 'M');
% fhM2 = plotSlice(data, pos, fsize, 'M', nchan);
% 
% 
% data = filter.W(25+2*49,:);
% data = processData(data, nchan);
% fhL1 = plotImg(data, pos, fsize, 'S');
% fhL2 = plotSlice(data, pos, fsize, 'S', nchan);

% if doprint
%     figname = 'whiten-filter-';
%     
%     printFig(fhS1, [figname 'L']);
%     printFig(fhS2, [figname 'Lcut']);
%     
%     printFig(fhM1, [figname 'M']);
%     printFig(fhM2, [figname 'Mcut']);
%     
%     printFig(fhL1, [figname 'S']);
%     printFig(fhL2, [figname 'Scut']);
% end

end

function printFig(fh, figname)
fsize = 4.5;

set(fh, 'PaperUnits', 'centimeters', 'PaperSize', [fsize fsize], 'PaperPosition', [0, 0, fsize, fsize])
print(fh, '-depsc2', '-r300', '-loose', figname)

end


function [fh] = plotSlice(data, pos, fsize, title, nchan)
fh = figure('Name', title, 'Color', 'w', 'Position', horzcat(pos, fsize));
hold on;

if nchan == 1
    colors = ['k'];
else
    colors = ['r', 'g', 'b'];
end

for ch = 1:nchan
    plot(data(4,:,ch), colors(ch))
end

xlim([1 7])
ylim([-1 1])

end

function [fh] = plotImg(data, pos, fsize, title, nchan)

fh = figure('Name', title, 'Color', 'w', 'Position', horzcat(pos, fsize));
data = 0.5+0.5*data;
imagesc(data)
if nchan == 1
   colormap('gray'); 
end
axis equal tight image off
set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])

end