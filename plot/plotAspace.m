function [ hf_chrom ] = plotAspace(A)

hf_chrom = figure('Name', ['A in DKL: ', A.name], 'Position', [0, 0, 1200, 800]);
set(0,'CurrentFigure', hf_chrom);
hold on;

[L, M] = size(A.rgb);

dkl = A.dkl;

pos = zeros(M, 4);

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
        
    %fprintf('%f %f \t| %f %f \t| %f %f\n', x0, x1, y0, y1, psize, n);

    %scatter(mx, my, 10*log10(n*20));
    line([x0, x1], [y0, y1], 'LineWidth', psize, 'Color', c);

end

l = 1.15*max(max(abs(pos)));

xlim([-l l])
ylim([-l l])

line([0, 0], [-l l], 'Color', [.8 .8 .8]); 
line([-l, l], [0 0], 'Color', [.8 .8 .8]);

end

