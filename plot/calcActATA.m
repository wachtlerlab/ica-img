function [ Act ] = calcActATA( Act )

Model  = Act.Model;
ds     = Model.ds;
ps     = double(ds.patchsize);
cfg    = Model.cfg;
nimg   = size(ds.imgdata, 4);
isize  = Act.isize;
offset = Act.offset;
imgset = createImageSet(cfg.data);
imgs   = imgset.images;

orgpatch = cell(nimg, 1);

for imgnr=1:nimg
    imgsml = permute(imgs{imgnr}.sml, [3 2 1]);
    csize = [size(imgsml, 2); size(imgsml, 2)];
    dsize = csize - isize(:, imgnr);
    dsize = (dsize / 2)';
    
    idx = imgallindicies(isize(1, imgnr), isize(2, imgnr), ps, 1);
    idx = idx + repmat(dsize, length(idx), 1);
    orgpatch{imgnr} = getpatches (imgsml, idx, ps)';
end

perimg = @(data, idx, func) cell2mat(arrayfun(...
    @(cimg) func(data(idx(cimg, 1):idx(cimg, 2), :))', 1:length(idx), ...
    'UniformOutput', false));

orgall = cell2mat(orgpatch);
clear orgpatch;
oL   = size(orgall, 2);


orgall = orgall - repmat(mean(orgall), size(orgall, 1), 1);
orgall = orgall / std(orgall(:)); %FIXME ?

nbf = size(Act.w, 2);
fprintf ('BF: ');

bf_mact = zeros(oL, nbf, nimg);

for n=1:nbf
    
    fprintf(' %03d', n);
    
    w = Act.w(:, n);
    if isfield(Act, 'wfilter')
       w = w .* Act.wfilter(:, n); 
    end
    
    wpatch  = w(:, ones(oL, 1)) .* orgall;
   
    mact = perimg(wpatch, offset, @mean);
    bf_mact(:, n, :) = mact;
    
    fprintf('\b\b\b\b');
end
fprintf (' all done.\n');
Act.mact = bf_mact;

end

function [patches] = getpatches(data, idx, patchsize)

 patches = patchesFromImg(data, idx, patchsize);
 patches = patches - mean(patches(:));
 patches = patches / sqrt(var(patches(:)));
    
end
