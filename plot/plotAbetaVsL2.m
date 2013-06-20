function [ fig ] = plotAbetaVsL2 (A)

fig = plotACreateFig(A, 'Beta vs L2', 0, [300 300]);
scatter(A.norm, A.beta, 'k+');
xlim([-1 max(A.norm)*1.1]);
xlabel('L^2 norm')
ylabel('\beta')
end

