function [ fh ] = plotImgStats(cfg)

if ischar(cfg)
  cfg = loadConfig (cfg, getDefaults('ica.configdir'));
end

[imgset, filter] = createImageSet (cfg.data);

nch  = imgset.shape(1); 
nx   = imgset.shape(2);
ny   = imgset.shape(3);
nimg = imgset.shape(4);

data = zeros(nch, nx*ny, nimg);
nbins = 40;


for n=1:nimg
    for c=1:nch
        data(c, :, n) = imgset.images{1}.data(c, :);   
    end
end

vals = zeros(nbins, nch, 2);
for c=1:nch
    X = data(c,:);
    fprintf('%s : mean: %5.3f, min: %5.3f, max %5.3f [%5.3f]\n', ...
        chan2str(filter.channels(c)), mean(X), min(X), max(X), max(X) - min(X));
    [nelms, elcnt] = hist(data(c,:), nbins);
    vals(:, c, 2) = nelms;
    vals(:, c, 1) = elcnt;
end


fh = plotACreateFig(cfg.id, 'Image pixel stats', 0, [400 600]);
[M, N] = plotCreateGrid(nch);
max_val = max(max(vals(:, :, 2)));
for c=1:nch
   subplot(N, M, c);
   %hist(data(c, :));
   bar(vals(:, c, 1),vals(:, c, 2), 'k')
   xlabel(chan2str(filter.channels(c)))
   ylim([0 max_val])
end

end

