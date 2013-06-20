function [ hf_chrom ] = plotAChroma(A, landscape)

if ~exist('landscape','var'); landscape = 0; end;

ps = A.ps;
[L, M] = size(A.rgb);
[nrows, ncols] = plotCreateGrid(M);

nrows = nrows+1;

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
    line([0, 0], [-0.8 0.8], 'Color', [.8 .8 .8]); 
    line([-0.8, 0.8], [0 0], 'Color', [.8 .8 .8]);
    scatter (x, y, 10, slice, 'filled');
    %axis equal off;
    axis equal off;
    axis ([-0.8 0.8 -0.8 0.8])

    %axis image;
end

for idx=M+1:length(hb)
    set (gcf, 'CurrentAxes', hb(idx));
    axis image off;
end

bpos = ncols*nrows-2;
pos = get(hb(bpos), 'Position');
tpos = (nrows-1)*ncols-1;
posup = get(hb(tpos), 'Position');

left = pos(1);
bottom = pos(2)+0.02;
width = (posup(1)-pos(1)+posup(3));
height = (posup(2)-pos(2)+posup(4))-0.03;
ax = axes('Units','normalized', ...
            'Position',[left bottom width height], ...
            'XTickLabel','', ...
            'YTickLabel','');
genLMSCS(0.1, ax);

end

