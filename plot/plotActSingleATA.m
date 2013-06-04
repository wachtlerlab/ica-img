function [ fig ] = plotActSingleATA( Act, bfnr, imgnr )

Model = Act.Model;
ds     = Model.ds;
ps     = double(ds.patchsize);
nchan  = 3;

fig = plotACreateFig(Model.id(1:7), 'Act Single ATA', 0, [300 300]);
load('colormaps_ck')

omraw = Act.mact(:, bfnr, imgnr);
lms = flipdim(reshape(omraw, nchan, ps, ps), 1);
rgb = permute(0.5 + 0.5 * (lms/max(abs(lms(:)))), [3 2 1]);
imagesc(rgb);
axis tight image off
set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])

end

