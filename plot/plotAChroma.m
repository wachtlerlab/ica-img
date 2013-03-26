function [ hf_chrom ] = plotAChroma(A, landscape)

if ~exist('landscape','var'); landscape = 0; end;

ps = A.ps;
[L, M] = size(A.rgb);
[nrows, ncols] = plotCreateGrid(M);


hf_chrom = plotACreateFig(A, 'Chroma', landscape, [1200, 800]);
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

for idx=M+1:length(hb)
    set (gcf, 'CurrentAxes', hb(idx));
    axis image off;
end

end

