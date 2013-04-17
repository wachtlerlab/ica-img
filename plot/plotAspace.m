function [ fig ] = plotAspace(A)

fig = plotACreateFig(A, 'A in DKL', 0, [1200, 600]);


[L, M] = size(A.rgb);
dkl = A.dkl;
pos = zeros(M, 4);

h1 = subplot (1, 2, 1);
hold on;

h2 = subplot (1, 2, 2);
hold on;

for idx=1:M
    pcs = A.pcs(:, idx)*5;
         
    x = dkl(:, 1, idx);
    y = dkl(:, 2, idx);
    z = dkl(:, 3, idx);
    
    mx = mean(x);
    my = mean(y);

    n = norm(A.sorted(:, idx));
    
    x0 = mx + pcs(1)/2;
    x1 = mx - pcs(1)/2;
    
    y0 = my + pcs(2)/2;
    y1 = my - pcs(2)/2;
    
    pos(idx, :) = [x0 x1 y0 y1];
    
    c = mean(reshape(A.rgb(:,idx), 3, 7*7), 2)';
    psize = 2*log10(n*25);

    set (gcf, 'CurrentAxes', h1);
    line([x0, x1], [y0, y1], 'LineWidth', psize, 'Color', c);
    
    set (gcf, 'CurrentAxes', h2);
    scatter(mx, my, psize*10, c, 'fill');
end

l = 1.15*max(max(abs(pos)));
set_layout(h1, l);
set_layout(h2, l);


end

function set_layout(h, l)
set (gcf, 'CurrentAxes', h);
xlim([-l l])
ylim([-l l])

line([0, 0], [-l l], 'Color', [.8 .8 .8]); 
line([-l, l], [0 0], 'Color', [.8 .8 .8]);

end