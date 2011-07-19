function plotAxis (A, bfs_selector)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

if isa (A, 'struct')
    name = A.name;
    A = A.A;

else
    name = 'direct';
end

[L,M] = size(A);

if nargin < 2
    bfs_selector = 1:M;
end

figure ('Name', ['Chromaticities of the basis functions: ', name]);
set (gcf,'Color',[0.9 0.9 0.9])

A = sortAbf (A);
B = reshapeAbf (A, 0);

n = length(bfs_selector) + 1;
ha = tight_subplot (n, 0,  [.01 .03], [.01 .01]);

for i=bfs_selector
    idx = 1 + (i - bfs_selector(1));
    hold on
    slice = squeeze (B(i,:,:,:));
    slice = reshape (slice, 49, 3);
    plotPatch (slice, ha(idx));
    text (0.5, 0.5, num2str (i));
end

genLMSCS (1/16, ha(n));

end


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