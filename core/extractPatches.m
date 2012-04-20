function [ patches ] = extractPatches(dataset, cluster)

[~, ~, ~, nimages] = size(dataset.imgdata);

xx = [];

for n = 1:nimages
  
  imgdata = dataset.imgdata(:,:,:,n);
  idxs = dataset.indicies(:,:, cluster, n);
  
  xtmp = patchesFromImg(imgdata, idxs, dataset.patchsize);
  xtmp = xtmp - mean(xtmp(:));
  xtmp = xtmp/sqrt(var(xtmp(:)));
  xx = [ xx xtmp ];
        
end


permidxlst = dataset.patsperm(:,cluster);
npats = dataset.npats;
x = xx(:,permidxlst);
x = x - (mean(x,2) * ones(1, npats));
patches = x/sqrt(var(x(:)));

end


function [ D ] = patchesFromImg(imgdata, indicies, patchsize)

[ndim, ~, ~] = size(imgdata);
N = patchsize^2;
ergvecdim = N * ndim;

npats = length(indicies); 

D = zeros (ergvecdim, npats);

for j = 1:npats,
  ix = indicies (j,1); 
  iy = indicies (j,2); 
  datapatch = imgdata (:, ix:ix+patchsize-1, iy:iy+patchsize-1);
  D(:,j) = reshape (datapatch, ergvecdim, 1);
end

end