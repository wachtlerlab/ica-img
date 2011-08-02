function plotPatch (basisFunction, axis_handle)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

if nargin > 1
    set (gcf, 'CurrentAxes', axis_handle);
end

colord = 0.5 + 0.5 * basisFunction / max(abs(basisFunction(:))); 
colord = flipdim (colord, 2);

slice = 0.5 + 0.5 * basisFunction / max(abs(basisFunction(:))); 

S = squeeze (slice(:,1));
M = squeeze (slice(:,2));
L = squeeze (slice(:,3));

x = L-M;
y = S-((L+M)/2);

hold on
scatter (x, y, 20, colord, 'filled');
axis ([-0.6 0.6 -0.6 0.6])
%axis equal
axis off

end
