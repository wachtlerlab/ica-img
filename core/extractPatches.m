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
