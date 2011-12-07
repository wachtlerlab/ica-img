function [ dataset ] = generateDataSet (images, fitPar, dataPar)
%GENERATEDATASET generate a dataset from a set of images

npats     = fitPar.npats;
nimages   = length(images);
patchsize = dataPar.patchSize;
blocksize = fitPar.blocksize;

nblocks = npats / blocksize;
nclusters = fitPar.maxIters / nblocks;
Tperimg = npats / nimages;

indicies = zeros(Tperimg, 2, nclusters, nimages, 'uint16');

for n = 1:nimages
  img = images(n);
  refBase = calcRefBase(img, patchsize);
  idx = generatePatchIndices(refBase, Tperimg, nclusters);
  indicies(:,:,:,n) = idx;  
end

patsperm = zeros(npats, nclusters);
for n = 1:nclusters
  patsperm(:,n) = randperm(npats);
end

dataset.images    = images;
dataset.patchsize = patchsize;
dataset.npats     = npats;
dataset.blocksize = blocksize;
dataset.nclusters = nclusters;
dataset.indicies  = indicies;
dataset.patsperm  = patsperm;

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