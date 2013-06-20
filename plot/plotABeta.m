function [ fig ] = plotABeta(A, sortbeta)

if ~exist('sortbeta', 'var'); sortbeta = 0; end;

color = [0 0 0];
[~,M] = size(A.data);

fig = plotACreateFig(A, 'Stats', 0, [300 300]);

if ~sortbeta
    val = A.beta;
else    
    val = sort(A.beta);
end

bar(1:M, val, 0.95, 'EdgeColor',color, 'FaceColor', color)
xlabel('BF index')
ylabel('$$\beta$$', 'interpreter', 'latex', 'fontsize', 12)
box off;
xlim([1 M])
ylim([0 10])

end

