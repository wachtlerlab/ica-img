function [ fh ] = plotActLM(cfg)

if ischar(cfg)
  cfg = loadConfig (cfg, getDefaults('ica.configdir'));
end

if (isstruct(cfg) && isfield(cfg, 'cfg'))
    cfg = cfg.cfg;
end


[imgset, ~] = createImageSet (cfg.data);

nchan = imgset.shape(1); 
nimg = imgset.shape(4);

imgdata = zeros(imgset.shape);
for n = 1:nimg
    img = imgset.images{n};
    imgdata(:,:,:,n) = img.data;
end

channels = imgset.channels;
N=1000;
fh = figure;
if nchan == 3
    lch = str2chan('L');
    mch = str2chan('M');
    ldata = imgdata(lch, :);
    mdata = imgdata(mch, :);
    
    p = randperm(length(ldata));
    idx = p(1:N);
    
    scatter(ldata(idx), mdata(idx), '+', 'k');
    xlabel('L')
    ylabel('M')
end

if nchan == 6
   lpch = find (channels == str2chan('L+'), 1);
   mpch = find (channels == str2chan('M+'), 1);
    
   lmch = find (channels == str2chan('L-'), 1);
   mmch = find (channels == str2chan('M-'), 1);
   
   lpdata = imgdata(lpch, :);
   mpdata = imgdata(mpch, :);
    
   lmdata = imgdata(lmch, :);
   mmdata = imgdata(mmch, :);
   
   p = randperm(length(lpdata));
   idx = p(1:N);
   
   subplot(1,2,1);
   scatter(lpdata(idx), mpdata(idx), '+', 'k');
   xlabel('L+')
   ylabel('M+')
   
   subplot(1,2, 2);
   scatter(lmdata(idx), mmdata(idx), '+', 'k');
   xlabel('L-')
   ylabel('M-')
end

end

