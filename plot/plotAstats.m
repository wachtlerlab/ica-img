function [fig] = plotAstats(A)

fig = plotACreateFig(A, 'Stats', 0, [900 300]);

color = [0 .5 .5];
[~,M] = size(A.data);

subplot(1, 2, 1)
bar(1:M, A.norm, 0.95, 'EdgeColor',color, 'FaceColor', color)
title('L2 norm')
xlabel('BF index')
ylabel('L2 norm')
box off;
xlim([0 M])

subplot(1, 2, 2)
bar(1:M, A.beta, 0.95, 'EdgeColor',color, 'FaceColor', color)
title('$$\beta$$ for BF', 'interpreter','latex', 'fontsize', 14)
xlabel('BF index')
ylabel('$$\beta$$', 'interpreter','latex', 'fontsize', 12)
box off;
xlim([1 M])

end

