function [fig] = plotAKurt(A, doSort)

if ~exist('doSort', 'var'); doSort = 0; end;

color = [0 0 0];
[~,M] = size(A.data);

fig = plotACreateFig(A, 'Stats', 0, [300 300]);

if ~doSort
    val = A.kurt;
else    
    val = sort(A.kurt);
end

bar(1:M, val, 0.95, 'EdgeColor',color, 'FaceColor', color)
xlabel('BF index')
ylabel('kurtosis')
box off;
xlim([1 M])