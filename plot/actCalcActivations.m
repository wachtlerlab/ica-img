function [ Act ] = actCalcActivations( Model )

ds  = Model.ds;
A = sortAbf (Model.A);

[~, L] = size(A);
W = pinv(A);


patchsize = double(ds.patchsize);
imgdata_ds = ds.imgdata;

[~, m, n, nimg ] = size(imgdata_ds);

idx = imgallindicies(m, n, patchsize, 1);


fprintf('Generating patchset\n');
bf_w = zeros(size(idx,1), nimg, L);
for imgnr=1:nimg %fixme
    fprintf(' %02d ', imgnr);
    imgdata = imgdata_ds(:,:,:,imgnr);   
    patches = getpatches (imgdata, idx, patchsize)';
    
    kall = size(patches, 1);
    for bfi=1:L
        bf = W(bfi,:);
        fprintf(' %03d', bfi);
        w = sum(bf(ones(kall, 1),:) .* patches, 2);
        fprintf('\b\b\b\b');
        bf_w(:, imgnr, bfi) = w;

    end
    
    fprintf('\b\b\b\b');
end


Act.Model = Model;
Act.idx = idx;
Act.w = bf_w;

end

function [patches] = getpatches(data, idx, patchsize)

 patches = patchesFromImg(data, idx, patchsize);
 patches = patches - mean(patches(:));
 patches = patches / sqrt(var(patches(:)));
    
end