function [ model ] = cpca(cfg)

currev = getCurRev ();
cfg = icaConfigLoad(cfg);

fprintf ('Starting simulation for %s [code: %s]\n', cfg.id, currev);

imageset = createImageSet (cfg.data);

tstart = tic;
fprintf('\nGenerating dataset...\n');
dataset = createDataSet (imageset, cfg);
fprintf('   done in %f\n', toc(tstart));

dim = dataset.dim;
cfgid = cfg.id;

model.ds = dataset.id;
model.cfg = cfgid;
model.creator = genCreatorId();
model.ctime = gen_ctime();

id = DataHash (model, struct ('Method', 'SHA-1'));

model.cfg = cfg;
model.ds = dataset;

model = setfield_idx(model, 'id', id, 1);



%patches = extractPatches(dataset, 1);
%patches = patches';
patches = genall(dataset);

fprintf('performing PCA\n')
C = cov(patches);
[U, D] = svd(C);

model.A = U;

end

function [patches] = genall(ds)

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

icaall = cell2mat(icapatch);

clear icapatch orgpatch iidx;
kall = size(icaall, 1);

icaall = icaall - repmat(mean(icaall), kall,  1);
patches = icaall / std(icaall(:));

end

function [patches] = getpatches(data, idx, patchsize)

 patches = patchesFromImg(data, idx, patchsize);
 patches = patches - mean(patches(:));
 patches = patches / sqrt(var(patches(:)));
    
end