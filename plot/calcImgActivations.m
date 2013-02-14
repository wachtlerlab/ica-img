function [Act] = calcImgActivations(Model)

ds  = Model.ds;
M = sortModelA(Model);

[~, L] = size(M.A);
W = pinv(M.A);

patchsize = double(ds.patchsize);

nimg = size(ds.imgdata, 4);

icapatch = cell(nimg, 1);

imgdata_ds = ds.imgdata;
isize = zeros(2, nimg);

fprintf('Generating patchset\n');
for imgnr=1:nimg
    %imgnr = 2;
    fprintf(' %02d', imgnr);
    imgdata = imgdata_ds(:,:,:,imgnr);
    
    [~, m, n] = size(imgdata);
    idx = imgallindicies(m, n, patchsize, 1);
    icapatch{imgnr} = getpatches (imgdata, idx, patchsize)';
    isize(:, imgnr) = [m; n];
    
    fprintf('\b\b\b');
end

npatches = cellfun(@(x) size(x, 1), icapatch);
sumpatch = cumsum(npatches);
ioffsets = [sumpatch - (npatches(1)-1), sumpatch];

icaall = cell2mat(icapatch);

clear icapatch orgpatch;
kall = size(icaall, 1);

icaall = icaall - repmat(mean(icaall), kall,  1);
icaall = icaall / std(icaall(:));

bf_w = zeros(kall, L);

fprintf('Calc patch activations\n');

for bfi=1:L
    bf = W(bfi,:);
    fprintf(' %03d', bfi);
    w = sum(bf(ones(kall, 1),:) .* icaall, 2);
    fprintf('\b\b\b\b');
    bf_w(:, bfi) = w;
end

Act.Model = Model;
Act.w = bf_w;
Act.offset = ioffsets;
Act.isize = isize;

end


function [patches] = getpatches(data, idx, patchsize)

 patches = patchesFromImg(data, idx, patchsize);
 patches = patches - mean(patches(:));
 patches = patches / sqrt(var(patches(:)));
    
end
