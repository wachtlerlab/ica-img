function actHistBf (Act, bf)

[N, nimg, nbf] = size(Act.w);

figure('Name', ['BF: ' num2str(bf)]);

[~, idx] = sortAbf(Act.Model.A);
[gm, gn] = plotCreateGrid(nimg);

allval = Act.w(:, :, idx(bf));
xmin = min(allval(:));
xmax = max(allval(:));
for img=1:nimg
    subplot(gm, gn, img);
    hold on;
    data = Act.w(:, img, idx(bf));
    n = hist(data, 80);
    hist(data, 80);
    xlim([xmin, xmax])
    me = mean(data)
    plot([me, me], [0, max(n)*1.01], 'r--');
end

end