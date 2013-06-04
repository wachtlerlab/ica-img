function [figL2, figBeta] = plotAstats(A)

figL2 = plotACreateFig(A, 'Stats', 0, [300 300]);

%color = [0 .5 .5];
color = [0 0 0];
[~,M] = size(A.data);

bar(1:M, A.norm, 0.95, 'EdgeColor',color, 'FaceColor', color)
xlabel('BF index')
ylabel('L2 norm')
box off;
xlim([0 M])

figBeta = plotACreateFig(A, 'Stats', 0, [300 300]);
bar(1:M, A.beta, 0.95, 'EdgeColor',color, 'FaceColor', color)
xlabel('BF index')
ylabel('$$\beta$$', 'interpreter', 'latex', 'fontsize', 12)
box off;
xlim([1 M])
ylim([0 10])

end

