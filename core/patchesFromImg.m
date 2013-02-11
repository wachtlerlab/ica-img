function [ D ] = patchesFromImg(imgdata, indicies, patchsize)

[ndim, ~, ~] = size(imgdata);
N = patchsize^2;
ergvecdim = N * ndim;

indicies = uint16(indicies);
patchsize = uint16(patchsize);

npats = length(indicies); 

D = zeros (ergvecdim, npats);

for j = 1:npats,
  ix = indicies (j,1); 
  iy = indicies (j,2); 
  datapatch = imgdata (:, ix:ix+patchsize-1, iy:iy+patchsize-1);
  D(:,j) = reshape (datapatch, ergvecdim, 1);
end

end