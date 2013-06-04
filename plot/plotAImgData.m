function [ fig ] = plotAImgData(Model, imgnr)
fig = plotACreateFig(Model.id(1:7), 'Act Single Img', 0, [300 300]);

cfg    = Model.cfg;

imgset = createImageSet(cfg.data);
imgs   = imgset.images;

imgsml = imgs{imgnr}.sml;
imagesc(mean(imgsml, 3))
colormap('gray')
axis tight image off;
set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])


end

