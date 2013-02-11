function [ hf_chrom ] = plotAChroma(A)

ps = 7;
[L, M] = size(A.rgb);
[nrows, ncols] = plotCreateGrid(M);

hold off;
hf_chrom = figure('Name', ['Chrom: ', A.name], 'Position', [0, 0, 1200, 800]);
set(0,'CurrentFigure', hf_chrom);
hb = tight_subplot(nrows, ncols, [.01 .01], [.01 .01]);

%pcs = zeros (2, M);
for idx=1:M

    set (gcf, 'CurrentAxes', hb(idx));
    hold on
   
    slice = permute(reshape(A.rgb(:, idx), 3, ps*ps), [2 1]);
    
    x = A.dkl(:, 1, idx);
    y = A.dkl(:, 2, idx);
    
    % [pc,~,latent,~] = princomp([x y]);
    % pc = pc(:,1) * latent (1);
    % pcs(:,idx) = pc;
    
    hold on
    scatter (x, y, 20, slice, 'filled');
    %axis equal off;
    axis equal;
    axis ([-1 1 -1 1])
    %axis image;
end

end

