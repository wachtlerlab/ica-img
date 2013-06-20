function [ filter ] = Whitening (cfg)

if ~isfield(cfg, 'log')
  cfg.log = 0;
end


if isfield(cfg, 'gray')
   error('legacy config file detected')
end

if ~isfield(cfg, 'perchan')
   cfg.perchan  = 0;
end

filter.name = 'Whitening';
filter.setup = @WhiteningFilterSetup;
filter.function = @WhiteningFilterImage;
filter.log = cfg.log;
filter.rectify = cfg.rectify;
filter.chans = [str2chan('S') str2chan('M') str2chan('L')];
filter.patchsize = cfg.patchsize;
filter.perchan = cfg.perchan;

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
patchsize = this.patchsize;
psq = patchsize^2;

img = cell(nimages, 1);
for n=1:nimages
    fprintf ('\t%s\n', images{n}.filename);
    img{n} = smlfilter.function (smlfilter, images{n});
end

n = img{1}.edgeN;
m = img{1}.edgeN;

pos =  imggencoords(n, m, patchsize, 1);

npats = length(pos);
nchan = length(channel);

if this.perchan
    data = zeros(psq, nimages*npats, nchan);
else
    data = zeros(psq*nchan, nimages*npats);
end

for n=1:nimages
  fprintf ('\t%s\n', images{n}.filename);
  
  st = (n-1)*npats + 1;
  ed = n*npats;
  
  if this.perchan
      for ch = channel
         chdata = imgallpatches(img{n}.sml(:,:,ch), patchsize);
         data(:, st:ed, ch) = chdata;
      end
  else
      imgdata = imgallpatches(img{n}.sml(:,:,channel), patchsize);
      data(:, st:ed) = imgdata;
  end
  
end

fprintf ('\n\t[W]\n');
if this.perchan
    W = zeros(psq, psq, length(channel));
    for ch = channel
        W(:,:,ch) = whiten_filter(data(:, :, ch));
    end
    this.W = W;
else
    this.W = whiten_filter(data);
end
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

if this.perchan
    Y = zeros(patchsize, patchsize, c, length(pos));
    
    for ch = 1:c
        patches = imgallpatches(input(:,:,ch), patchsize, patchsize);
        X = this.W(:,:,ch) * patches;
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
