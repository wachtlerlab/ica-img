function [ dataset ] = createDataSet (imageset, cfg, shuffle)
%GENERATEDATASET generate a dataset from a set of images
%                A dataset is supposed to contain all *generated* data
%                neccessary to instannciate a analysis and run it

if nargin > 2 && shuffle 
  rng('shuffle');
end

creator = genCreatorId();
cfgid   = cfg.id;

rng_save = rng;
id       = generateId (cfgid, rng_save);

dcfg = cfg.data;

npats     = dcfg.npats;
patchsize = dcfg.patchsize;
blocksize = dcfg.blocksize;
nclusters = dcfg.ncluster;
nimages   = length(imageset.images);
nblocks   = npats / blocksize;
Tperimg   = npats / nimages;
maxiter   = nblocks*nclusters;

indicies = zeros(Tperimg, 2, nclusters, nimages, 'uint16');
imgdata = zeros(imageset.shape);

for n = 1:nimages
  img = imageset.images{n};
  refBase = calcRefBase(img, patchsize);
  idx = generatePatchIndices(refBase, Tperimg, nclusters);
  indicies(:,:,:,n) = idx;
  
  imgdata(:,:,:,n) = img.data;
end

patsperm = zeros(npats, nclusters, 'int32');
for n = 1:nclusters
  patsperm(:,n) = randperm(npats);
end

dim = patchsize * patchsize * imageset.shape(1);
Aguess = createMixingMatrix (cfg, dim);


dataset.id        = id;
dataset.cfg       = cfgid;
dataset.creator   = creator;
dataset.ctime     = gen_ctime();
dataset.rng       = rng_save;
dataset.dim       = dim;
dataset.Aguess    = Aguess;
dataset.imageset  = imageset;
dataset.imgdata   = imgdata;
dataset.channels  = imageset.channels;
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
[~, N, ~] = size (img.data);
N = N - patchsize;
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

%refBase format is now
% [x, y; 
%  x, y]

end

function [id] = generateId (cfgid, rng_settings)
s.cfgid = cfgid;
s.rng = rng_settings;

id = DataHash (s, struct ('Method', 'SHA-1'));
end
