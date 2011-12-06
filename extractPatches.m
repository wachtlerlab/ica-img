function [ patches ] = extractPatches(dataset, block)

nimages = length(dataset.images);

xx = [];

for n = 1:nimages
  
  img = dataset.images(n);
  idxs = dataset.indicies(:,:, block, n);
  
  xtmp = patchesFromImg(img, idxs, dataset.patchsize);
  xtmp = xtmp - mean(xtmp(:));
  xtmp = xtmp/sqrt(var(xtmp(:)));
  xx = [ xx xtmp ];
        
end

[~, T] = size(xx);
permidxlst = randperm(T);
x = xx(:,permidxlst);
x = x - (ones(T,1)*mean(x'))';
patches = x/sqrt(var(x(:)));

end


function [ D ] = patchesFromImg(img, indicies, patchsize)

[ndim, ~, ~] = size(img.imgData);
N = patchsize^2;
ergvecdim = N * ndim;

npats = length(indicies); 

D = zeros (ergvecdim, npats);

for j = 1:npats,
  ix = indicies (j,1); 
  iy = indicies (j,2); 
  datapatch = img.imgData (:, ix:ix+patchsize-1, iy:iy+patchsize-1);
  D(:,j) = reshape (datapatch, ergvecdim, 1);
end

end