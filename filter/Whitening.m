function [ filter ] = Whitening (cfg)

if ~isfield(cfg, 'log')
  cfg.log = 0;
end

if ~isfield(cfg, 'gray')
   cfg.gray  = 0;
end

filter.name = 'Whitening';
filter.setup = @WhiteningFilterSetup;
filter.function = @WhiteningFilterImage;
filter.log = cfg.log;
filter.rectify = cfg.rectify;
filter.chans = [str2chan('S') str2chan('M') str2chan('L')]; ;
filter.patchsize = cfg.patchsize;
filter.gray = cfg.gray;

smlcfg.log = 0;
filter.sml = SML(smlcfg);
filter.channels = mapChannel (filter.chans, filter.rectify);

end

function [this] = WhiteningFilterSetup (this, images)
fprintf ('Generating whitening matrix [W]\n');
nimages = length(images);

smlcfg.log = 0;
smlfilter = SML(smlcfg);
channel = this.chans;
data = [];
for n=1:nimages
  fprintf ('\t%s\n', images{n}.filename);
  img = smlfilter.function (smlfilter, images{n});
  if this.gray
      imgdata = [];
      for ch = channel
         chdata = imgallpatches(img.sml(:,:,ch), this.patchsize);
         imgdata = horzcat (imgdata, chdata);
      end
  else
      imgdata = imgallpatches(img.sml(:,:,channel), this.patchsize);
  end
  
  data = horzcat (data, imgdata);
end

fprintf ('\n\t[W]\n');
this.W = whiten_filter(data);
fprintf ('done.\n');
end

function [img] = WhiteningFilterImage (this, img)

sml_filter = this.sml;
img = sml_filter.function (sml_filter, img);

patchsize = this.patchsize;
channel = chanlist2idx(this.chans);
input = img.sml(:,:, channel);
[n, m, c] = size (input);

n = n - mod(n, patchsize);
m = m - mod(m, patchsize);
input = input(1:n,1:m,:);


pos = imggencoords(n, m, patchsize, patchsize);

if this.gray
    Y = zeros(patchsize, patchsize, c, length(pos));
    
    for ch = 1:c
        patches = imgallpatches(input(:,:,ch), patchsize, patchsize);
        X = this.W * patches;
        Y(:, :, ch, :) = reshape(X, patchsize, patchsize, length(pos));
    end
    
else
    
    patches = imgallpatches(input, patchsize, patchsize);
    X = this.W * patches;
    Y = reshape(X, patchsize, patchsize, c, length(pos));
end

android = zeros(n, m, c);
for idx = 1:length(pos)
   cr = pos(1, idx);
   cc = pos(2, idx);
   android(cr:cr+patchsize-1, cc:cc+patchsize-1, :) = Y(:,:,:,idx); %reshape(X(:, idx), patchsize, patchsize, c);  
end

data = android;
%di = data-android;
%max(di(:))


if this.rectify
  
  rdata = zeros(2*c, m, n);
  for n=1:c
    [on, off] = rectifyData (data(:, :, n), this.log);
    rdata((n*2-1),:,:) = on;
    rdata(n*2,:,:) = off;
  end
  img.data = permute (rdata, [1 3 2]);
else
    
  img.data = permute (data, [3 2 1]);
end

end


function [on, off] = rectifyData(data, doLog)

if nargin < 2
  doLog = 0;
end

on = data;
off = data;

on(on < 0) = 0;
off(off > 0) = 0;
off = -1 * off;

if doLog
  on = log (on + doLog * max (on(:)));
  off = log (off + doLog * max (off(:)));
end

end
