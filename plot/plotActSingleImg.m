function [ fig ] = plotActSingleImg(Act, bfnr, imgnr)

Model = Act.Model;
wall   = Act.w(:, bfnr);
offset = Act.offset;

fig = plotACreateFig(Model.id(1:7), 'Act Single Img', 0, [300 300]);
load('colormaps_ck')

w = wall(offset(imgnr,1):offset(imgnr,2), :);
snorm = max(abs(wall(:)));
edge = sqrt(length(w));
bfw = reshape(w, edge, edge);
bfwn = 0.5 * (bfw/snorm);
colormap(bwr_cmp)
imagesc(bfwn');
axis tight image off
set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])


end

