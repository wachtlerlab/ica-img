function [ filter ] = Vanilla(cfg)

if nargin < 1 || ~isfield(cfg, 'log')
  log = 0;
else
  log = cfg.log;
end

if isfield (cfg, 'rectify')
    filter.rectify = cfg.rectify;
else
    filter.rectify = 0;
end

if isfield(cfg, 'center_img')
    filter.center_img = cfg.center_img;
else
    filter.center_img = 0;
end

filter.name = 'Vanilla';
filter.function = @VanillaFilterImage;
filter.log = log;
filter.channels = mapChannel (cfg.channels, filter.rectify);
filter.input = cfg.channels;

end

function [img] = VanillaFilterImage (this, img)

si = size(img.hs_data);

if length(si) < 3
    [~, T] = size (img.hs_data);
    edgeN = sqrt (T);
    
    if edgeN ~= round (edgeN)
        error ('Image data to square!');
    end
    
    img.edgeN = edgeN;
    sx = edgeN;
    sy = edgeN;
    raw = img.hs_data;
else
    sx = si(2);
    sy = si(3);
    img.edgeN = [sx, sy];
    raw = img.hs_data(:,:); 
end

N = length(this.input);
if this.rectify
   N = 2*N; 
end

if this.center_img == 1
  fprintf('Filter: Centering images\n');
  m = mean(raw(:));
  for n = 1:length(this.input)  
    raw(n, :) = raw(n, :) - m;
    fprintf('\t [%d]: %f [%f]\n', n, m, mean(raw(n,:)));
  end
end

if this.rectify
    data = zeros(N, edgeN*edgeN);
    for n=1:length(this.input)
        [on, off] = rectifyData (raw(n,:), this.log);
        data((n*2-1),:,:) = on;
        data(n*2,:,:) = off;  
    end
else
    if this.log
        data = log(raw);
    else
        data = raw;
    end
end

img.data  = reshape (data, N, sx, sy); %(c,x,y) [f,c,r]
img.sml = squeeze(permute (raw, [3 2 1])); % compat (y,x,c) [f, c, r]

end


function [on, off] = rectifyData(data, doLog)

if nargin < 2
  doLog = 0;
end

data = data - mean(data(:));

on = data;
off = data;

on(on < 0) = 0;
off(off > 0) = 0;
off = -1 * off;

offset = doLog * min(abs(data(:)));

if doLog
  on = log (on + offset);
  off = log (off + offset);
end

end