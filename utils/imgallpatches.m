function [data] = imgallpatches(img, patchsize)

[m, n, c] = size (img);

nm = m - patchsize;
nn = n - patchsize;

cr = repmat (1:nm, 1, nn);
cc = sort (cr);

samplesize = patchsize^2 * c;
npats = length(cc);
data = zeros (samplesize, npats);

for k=1:npats
  patch = img(cr(k):cr(k)+patchsize-1, cc(k):cc(k)+patchsize-1, :);
  data(:,k) = reshape (patch, samplesize, 1);
end

end

