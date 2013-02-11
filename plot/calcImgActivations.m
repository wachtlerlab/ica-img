function [totalact] =  calcImgActivations(Model, ds, imgset)

M = sortModelA(Model);

[~, L] = size(M.A);
W = pinv(M.A);

patchsize = double(ds.patchsize);

nimg = 8;
%per image

patchset = cell(nimg, 1);
imgset_img = imgset.images;
imgdata_ds = ds.imgdata;

parfor imgnr=1:nimg
    %imgnr = 2;
    imgdata = imgdata_ds(:,:,:,imgnr);
    
    [~, m, n] = size(imgdata);
    idx = imgallindicies(m, n, patchsize, 1);
    spatch_ = patchesFromImg(imgdata, idx, patchsize);
    spatch_ = spatch_ - mean(spatch_(:));
    spatch_ = spatch_ / sqrt(var(spatch_(:)));
    
    patchset{imgnr}.spatch = spatch_;
    
    imgsml = permute(imgset_img{imgnr}.sml, [3 2 1]);
    [~, mo, no] = size(imgsml);
    
    dm = (mo - m) / 2;
    dn = (no - n) / 2;
    idx(:, 1) = idx(:, 1) + dm;
    idx(:, 2) = idx(:, 2) + dn;
     
    opatch_ = patchesFromImg(imgsml, idx, patchsize);
    opatch_ = opatch_ - mean(opatch_(:));
    opatch_ = opatch_ / sqrt(var(opatch_(:)));
    
    patchset{imgnr}.opatch = opatch_;
end

%per img and bf
imgact = cell(nimg, 1);
fprintf('Calc patch activations\n');

parfor imgnr=1:nimg
    fprintf('\t img %d\n', imgnr);

    spatch = patchset{imgnr}.spatch;
    opatch = patchset{imgnr}.opatch;

    [~, k] = size(opatch);

    for bfi=1:L
        bf = W(bfi,:);
        
        bfr = repmat(bf', 1, k);        
        act = dot(bfr, spatch);
        
        smraw = calcpatchact(act, spatch);
        omraw = calcpatchact(act, opatch);
        
        imgact{imgnr}.bf{bfi}.smraw = smraw;
        imgact{imgnr}.bf{bfi}.omraw = omraw;
        imgact{imgnr}.bf{bfi}.w     = act;
        
    end
    
    imgact{imgnr}.spatches = spatch;
    imgact{imgnr}.opatches = opatch;
    
end

totalact = imgact;

end

function [mraw] = calcpatchact (imgact, patches)

raw  = repmat(imgact, size(patches, 1), 1) .* patches;
mraw = mean(raw, 2);
%msml = reshape(flipdim(reshape(mraw, 3, ps, ps), 1), bflen, 1);
%mrgb = 0.5 + 0.5 * (msml/max(abs(msml)));

end



