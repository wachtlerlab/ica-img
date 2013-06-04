function [icaall, idx, ioffsets] = allpatches(Model)

ds  = Model.ds;
A = sortAbf (Model.A);

[~, L] = size(A);
W = pinv(A);

patchsize = double(ds.patchsize);

nimg = size(ds.imgdata, 4);

icapatch = cell(nimg, 1);
iidx = cell(nimg, 1);

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
    iidx{imgnr} = idx;
    isize(:, imgnr) = [m; n];
    
    fprintf('\b\b\b');
end

npatches = cellfun(@(x) size(x, 1), icapatch);
sumpatch = cumsum(npatches);
ioffsets = [sumpatch - (npatches(1)-1), sumpatch];

icaall = cell2mat(icapatch);
idx = cell2mat(iidx);

clear icapatch orgpatch iidx;
kall = size(icaall, 1);

icaall = icaall - repmat(mean(icaall), kall,  1);
icaall = icaall / std(icaall(:));
end


function [patches] = getpatches(data, idx, patchsize)

 patches = patchesFromImg(data, idx, patchsize);
 patches = patches - mean(patches(:));
 patches = patches / sqrt(var(patches(:)));
    
end