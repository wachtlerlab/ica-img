function [Act] =  calcImgActivations(Model)

ds  = Model.ds;
cfg = Model.cfg;

imgset = createImageSet(cfg.data);

M = sortModelA(Model);

[~, L] = size(M.A);
W = pinv(M.A);

patchsize = double(ds.patchsize);

nimg = 8;

icapatch = cell(nimg, 1);
orgpatch = cell(nimg, 1);

patchset = cell(nimg, 1);

imgset_img = imgset.images;
imgdata_ds = ds.imgdata;


fprintf('Generating patchset\n');
for imgnr=1:nimg
    %imgnr = 2;
    fprintf(' %02d', imgnr);
    imgdata = imgdata_ds(:,:,:,imgnr);
    
    [~, m, n] = size(imgdata);
    idx = imgallindicies(m, n, patchsize, 1);
    icapatch{imgnr} = getpatches (imgdata, idx, patchsize)';
    fprintf('.');

    fprintf(':');
    imgsml = permute(imgset_img{imgnr}.sml, [3 2 1]);
    
    [~, mo, no] = size(imgsml);
    dm = (mo - m) / 2;
    dn = (no - n) / 2;
    idx(:, 1) = idx(:, 1) + dm;
    idx(:, 2) = idx(:, 2) + dn;
    
    orgpatch{imgnr} = getpatches (imgsml, idx, patchsize)';
    fprintf('\b\b\b\b\b');
end

npatches = cellfun(@(x) size(x, 1), icapatch);
sumpatch = cumsum(npatches);
ioffsets = [sumpatch - (npatches(1)-1), sumpatch];

icaall = cell2mat(icapatch);
orgall = cell2mat(orgpatch);

clear icapatch orgpatch;
oL   = size(orgall, 2);
kall = size(icaall, 1);

icaall = icaall - repmat(mean(icaall), kall,  1);
%orgall = orgall - repmat(mean(orgall, 2), 1, oL);

icaall = icaall / std(icaall(:));
%orgall = orgall / std(orgall(:));

perimg = @(data, idx, func) cell2mat(arrayfun(...
    @(cimg) func(data(idx(cimg, 1):idx(cimg, 2), :))', 1:length(idx), ...
    'UniformOutput', false));

bf_w = zeros(kall, L);
bf_mact = zeros(oL, L, nimg);

fprintf('Calc patch activations\n');

for bfi=1:L
    bf = W(bfi,:);

    fprintf(' %03d', bfi);
    
    w = sum(bf(ones(kall, 1),:) .* icaall, 2);

    fprintf('.');
    wpatch  = w(:, ones(oL, 1)) .* orgall;
    fprintf(':');
    mact = perimg(wpatch, ioffsets, @mean);
    fprintf('\b\b\b\b\b\b');
    %bfact{bfi} = struct('w', w, 'mop', mact);
    bf_w(:, bfi) = w;
    bf_mact(:, bfi, :) = mact;
end

Act.w = bf_w;
Act.mact = bf_mact;
Act.Model = Model;
Act.offset = ioffsets;

end

function [mraw] = calcpatchact (imgact, patches)

raw  = repmat(imgact, size(patches, 1), 1) .* patches;
mraw = mean(raw, 2);
%msml = reshape(flipdim(reshape(mraw, 3, ps, ps), 1), bflen, 1);
%mrgb = 0.5 + 0.5 * (msml/max(abs(msml)));

end


function [patches] = getpatches(data, idx, patchsize)

 patches = patchesFromImg(data, idx, patchsize);
 patches = patches - mean(patches(:));
 patches = patches / sqrt(var(patches(:)));
    
end
