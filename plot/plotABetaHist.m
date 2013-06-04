function [ fh ] = plotABetaHist(A)
fh = plotACreateFig(A, 'Beta Histogram', 0, [300 300]);

color = [0 0 0];
[n, c] = hist(A.beta, 10);
bar(c, n, 'EdgeColor', color, 'FaceColor', color);
xlim([0 10]);
ylim([0 120]);

end

