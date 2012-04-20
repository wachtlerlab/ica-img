function [ dataset ] = createDataSet (images, cfg)
%GENERATEDATASET generate a dataset from a set of images
%                A dataset is supposed to contain all *generated* data
%                neccessary to instannciate a analysis and run it

dcfg = cfg.data;

npats     = dcfg.npats;
patchsize = dcfg.patchsize;
blocksize = dcfg.blocksize;
nclusters = dcfg.ncluster;
nimages   = length(images);
nblocks   = npats / blocksize;
Tperimg   = npats / nimages;
maxiter   = nblocks*nclusters;

indicies = zeros(Tperimg, 2, nclusters, nimages, 'uint16');

[z, x, y] = size (images{1}.imgData);
imgdata = zeros(z, x, y, nimages);

for n = 1:nimages
  img = images{n};
  refBase = calcRefBase(img, patchsize);
  idx = generatePatchIndices(refBase, Tperimg, nclusters);
  indicies(:,:,:,n) = idx;
  
  data = reshape (img.imgData, z, x, y);
  imgdata(:,:,:,n) = data;
end

patsperm = zeros(npats, nclusters, 'int32');
for n = 1:nclusters
  patsperm(:,n) = randperm(npats);
end

dim = patchsize * patchsize * z;
Aguess = createMixingMatrix (cfg, dim);

dataset.dim       = dim;
dataset.Aguess    = Aguess;
dataset.images    = images;
dataset.imgdata   = imgdata;
dataset.patchsize = patchsize;
dataset.npats     = npats;
dataset.blocksize = blocksize;
dataset.nclusters = nclusters;
dataset.indicies  = indicies;
dataset.patsperm  = patsperm;
dataset.maxiter   = maxiter;


end


function [indices] = generatePatchIndices(refBase, nidx, len)

cnt = size (refBase, 1);

indices = zeros(nidx, 2, len);

for n = 1:len
  inneridxidx = randperm (cnt);
  indices(:,:,n) = refBase(inneridxidx(1:nidx), :);
end

end

function [refBase] = calcRefBase(img, patchSize)

% generate posible combinations of patch indices
patchsize = patchSize;
N = img.edgeN - patchsize;
reps = repmat (1:N, 1, N);
l = [sort(reps); reps];

refa = img.refkoos(1) - patchsize;
refb = img.refkoos(3);
refc = img.refkoos(2) - patchsize;
refd = img.refkoos(4);

%filter out coordinates of the ref card
validIdx = l(:, ~((l(1,:) > refa & l(1,:) < refb) & ...
  (l(2,:) > refc & l(2,:) < refd)));

refBase = permute (validIdx, [2 1]);
end

