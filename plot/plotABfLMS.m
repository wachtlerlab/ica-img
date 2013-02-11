function [hf] = plotABfLMS (Model, fh)

if isa (Model, 'struct')
    A = Model.A;
    name = Model.name;
else
    error ('Need model');
end

[~,M] = size(A);
A = sortAbf (A);
A = colorAbf (A);
B = reshapeAbf (A);

if nargin < 2
  hf = figure('Name', ['Basis Functions: ', name], 'Position', [0, 0, 800, 1000]);
else
  hf = fh;
  set (0, 'CurrentFigure', fh);
end

root = sqrt (M);
nrows = ceil(root);
ncols = floor(root);
ha = tight_subplot(nrows, ncols, [.01 .03], [.01 .01]);

for idx=1:M

    bf = squeeze (B(idx, :, :, :));
    set (gcf, 'CurrentAxes', ha(idx));
    %title (num2str (idx));
    hold on
    image(bf);
    axis image;
    axis off;
end

end

%%
function [ shapedA ] = reshapeAbf (A, doflip)
%reshapeAbf Reshape the mixing matrix A to obtain the basis functions

if isa (A, 'struct')
    A = A.A;
end

if ndims (A) ~= 2
   error ('Matrix must be 2D') 
end

if nargin < 2
    doflip = 1;
end

[m, n] = size (A);

c = sqrt (m/3);

if (c ~= round (c))
    error ('Basis function not like X*X*3')
end

if doflip
    A = flipdim (A, 1);
end

R = reshape (A, 3, c, c, n);
shapedA = permute (R, [4 3 2 1]);

end

%%
function [ colored ] = colorAbf (A, submin)

if isa (A, 'struct')
    A = A.A;
end

if ndims (A) > 2
   error ('Array or 2D-matrix expected') 
end

C = num2cell (A, 1);

if nargin >= 2 && submin
    C = cellfun (@subMin, C, 'UniformOutput', false);
end

C = cellfun (@normColor, C, 'UniformOutput', false);

colored = cell2mat (C);
end


function [B] = subMin (A)
B = A - min(A(:));
end


function [normed] = normColor (A)
normed = 0.5 + 0.5 * A / max(abs(A(:))); 
end