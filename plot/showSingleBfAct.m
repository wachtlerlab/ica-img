function [bfact_fig] = showSingleBfAct(Model, Act, bfnr)

nimg = length(Act);

nrows = 2;
ps    = 7;
nchan = 3;

ncols = nimg;

bfact_fig = figure('Name', ['Chrom: ', Model.id(1:7)], 'Position', [0, 0, 1200, 400]);

load('colormaps_ck')

snorm = 1.0;

for n=1:nimg
    w=Act{n}.bf{bfnr}.w;
    snorm = max(snorm, max(abs(w)));
end

for n=1:nimg
    bfact=Act{n}.bf{bfnr};

    pidx = getplotidx(ncols, 1, n);
    subplot(nrows, ncols, pidx);
    
    edge = sqrt(length(bfact.w));
    w = bfact.w;
    bfw = reshape(w, edge, edge);
    bfwn = 0.5 * (bfw/snorm);
    
    fprintf('img %d: min: %f mean: %f max: %f\n', ...
        n, min(w), mean(w), max(w));
    colormap(bwr_cmp)
    
    imagesc(bfwn');
    axis image off;
    
    pidx = getplotidx(ncols, 2, n);
    subplot(nrows, ncols, pidx);
    
    lms = flipdim(reshape(bfact.omraw, nchan, ps, ps), 1);
    rgb = permute(0.5 + 0.5 * (lms/max(abs(lms(:)))), [2 3 1]);
    imagesc(rgb);
    axis image off;
end

end

function idx = getplotidx(N, m, n)
    idx = N*(m-1)+n;
end