function plotMStats(model)

A = model.A;

[~,M] = size(A);

nA = ones (1, M);

for m = 1:M
  nA(m) = norm(A(:,m));
end

[~, idx] = sort(-nA);
sortedA = A (:,idx);

color = [0 .5 .5];
figtitle = ['Stats' model.cfg.id(1:7) '-' model.id(1:7)];
curfig = figure('Name', figtitle, 'Position', [0 0 900 300], ...
    'Color', 'w', 'PaperType', 'A4');

subplot(1, 2, 1)
bar(1:M, nA(idx), 0.95, 'EdgeColor',color, 'FaceColor', color)
title('L2 norm')
xlabel('BF index')
ylabel('L2 norm')
box off;
xlim([0 M])

subplot(1, 2, 2)
bar(1:M, model.beta(idx), 0.95, 'EdgeColor',color, 'FaceColor', color)
title('$$\beta$$ for BF', 'interpreter','latex', 'fontsize', 14)
xlabel('BF index')
ylabel('$$\beta$$', 'interpreter','latex', 'fontsize', 12)
box off;
xlim([1 M])

end

