function [ ftData ] = filterIDManual (imageData, dataPar)

s = size (imageData);

if s(1) ~= 3
   error ('Image data in bad format')
end

nPixel = s(2) * s(3);

dm = 2;
dl = 3;

mx = reshape (squeeze (imageData(dm, :, :)), 1, nPixel);
lx = reshape (squeeze (imageData(dl, :, :)), 1, nPixel);

mlx = mean ([mx; lx]);
ml = reshape (mlx, s(2), s(3));

for n=1:3
    ftData(n,:,:) = surFilter (imageData (n, :, :), ml, dataPar.filter);
end

end


function [out] = surFilter(data, surData, filter)

data = squeeze (data);
ml = squeeze (surData);

s = size (data);
out = zeros (s(1) - 2, s(2) - 2);

for m = 2:s(1)-1
    for n = 2:s(2)-1
        center = data (m, n);
        surround = ml (m-1:m+1, n-1:n+1);
        rf = surround;
        rf(2,2) = center;
        rf = rf .* filter;
        summed = sum (rf(:));
        out (m - 1, n - 1) = summed;
    end
end

end
