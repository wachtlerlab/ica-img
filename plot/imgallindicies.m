function [indicies] = imgallindicies(m, n, patchsize, tiling)

if ~exist('tiling', 'var'); tiling = 1; end

sm = gen_coords(m, patchsize, tiling);
sn = gen_coords(n, patchsize, tiling);

cm = repmat(sm, 1, length(sn));
cn = sort(repmat(sn, 1, length(sm)));

indicies = [cm; cn]';

end


function [y] = gen_coords(x, patchsize, tiling)

r = mod(x, tiling) - (patchsize - 1);
x = x + r * (r < 1);
y = 1:tiling:x;

end
