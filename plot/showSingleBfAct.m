function [bfact_fig] = showSingleBfAct(Act, bfnr)

Model = Act.Model;

ds     = Model.ds;
ps     = double(ds.patchsize);
cfg    = Model.cfg;
nimg   = size(ds.imgdata, 4);
nchan  = 3;
wall   = Act.w(:, bfnr);
offset = Act.offset;


nrows  = 3 + isfield(Act, 'mact') + isfield(Act, 'wfilter');
nrows  = nrows + isfield(Act, 'epp');

ncols = nimg;

bfact_fig = figure('Name', ['Chrom: ', Model.id(1:7)], 'Position', [0, 0, 1200, 400]);

load('colormaps_ck')

snorm = max(abs(wall(:)));

for n=1:nimg
    
    w = wall(offset(n,1):offset(n,2), :);
    
    pidx = getplotidx(ncols, 1, n);
    subplot(nrows, ncols, pidx);
    
    edge = sqrt(length(w));
    bfw = reshape(w, edge, edge);
    bfwn = 0.5 * (bfw/snorm);
    
    fprintf('img %d: min: %f mean: %f max: %f\n', ...
        n, min(w), mean(w), max(w));
    colormap(bwr_cmp)
    
    imagesc(bfwn');
    axis image off;
    
    pidx = getplotidx(ncols, 2, n);
    subplot(nrows, ncols, pidx);
    hist(w, 20)
    xlim([min(w), max(w)])
    crow = 3;  
        
    if isfield(Act, 'epp')
        pidx = getplotidx(ncols, crow, n);
        subplot(nrows, ncols, pidx);
        hold on;
        
        epp = Act.epp(bfnr, n, :);
        xrange = min(w):0.1:max(w);
        y = expwrpdf(xrange, epp(1), epp(2), epp(3));
        
        [nelm, xcnt] = hist(w, 20);
        [ax, h1, h2] = plotyy(xcnt, nelm, xrange, y, 'bar', 'plot');
       
        set(h1,'FaceColor',[0 .5 .5],'EdgeColor',[0 .5 .5])
        set(h2, 'LineWidth', 1, 'Color', 'r');
        
        crow = crow + 1;
        
    end
    
    if isfield(Act, 'wfilter')
        pidx = getplotidx(ncols, crow, n);
        subplot(nrows, ncols, pidx);
        wfilter = Act.wfilter(offset(n,1):offset(n,2), bfnr);
        
        wfiltered = w(wfilter);
        hist(wfiltered, 20)
        crow = crow + 1;
        xlim([min(w), max(w)])
    end

    if isfield(Act, 'mact')

        pidx = getplotidx(ncols, crow, n);
        subplot(nrows, ncols, pidx);
        
        omraw = Act.mact(:, bfnr, n);
        lms = flipdim(reshape(omraw, nchan, ps, ps), 1);
        rgb = permute(0.5 + 0.5 * (lms/max(abs(lms(:)))), [2 3 1]);
        imagesc(rgb);
        axis image off;
    
    end
    
end

pidx = getplotidx(ncols, nrows, 1);
pend = pidx+(nimg-(nimg/2));
subplot(nrows, ncols, pidx:pend);
hist(wall, 100)

if isfield(Act, 'mact')
    subplot(nrows, ncols, pend+2);
    omraw = mean(Act.mact(:, bfnr, :), 3);
    lms = flipdim(reshape(omraw, nchan, ps, ps), 1);
    rgb = permute(0.5 + 0.5 * (lms/max(abs(lms(:)))), [2 3 1]);
    imagesc(rgb);
    axis image off;
end

end

function idx = getplotidx(N, m, n)
    idx = N*(m-1)+n;
end