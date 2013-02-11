function [data, pos] = imgallpatches(img, patchsize, tiling)

if ~exist('tiling', 'var'); tiling = 1; end

[m, n, c] = size (img);

sm = gen_coords(m, patchsize, tiling);
sn = gen_coords(n, patchsize, tiling);

cm = repmat(sm, 1, length(sn));
cn = sort(repmat(sn, 1, length(sm)));

pos = [cm; cn];

samplesize = patchsize^2 * c;
npats = length(pos);
data = zeros (samplesize, npats);

for k=1:npats
  rm = pos(1,k):pos(1,k)+patchsize-1;
  rn = pos(2,k):pos(2,k)+patchsize-1;
  
  patch = img(rm, rn, :);
  data(:,k) = reshape (patch, samplesize, 1);
end

end


function [y] = gen_coords(x, patchsize, tiling)

r = mod(x, tiling) - (patchsize - 1);
x = x + r * (r < 1);
y = 1:tiling:x;

end


% r = mod(m, tiling) - (patchsize - 1);
% m = m + r * (r < 1);
% 
% r = mod(n, tiling) - (patchsize - 1);
% n = n + r * (r < 1);
% 
% sm = 1:tiling:m;
% sn = 1:tiling:n;

%r = mod(m, patchsize);
%m = m - max((patchsize * (tiling == 1)), r);
%n = n - max((patchsize * (tiling == 1)), r);
%cr = repmat (1:tiling:m, 1, n/tiling);
%cc = sort (cr);
%pos = [cr; cc];