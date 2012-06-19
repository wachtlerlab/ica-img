function [ filter ] = CSRect(cfg)
%CSRECT Summary of this function goes here
%   Detailed explanation goes here

if ~isfield(cfg, 'log')
  cfg.log = 0;
end

filter.name = 'CSRect';
filter.function = @CSRectFilterImage;
filter.log = cfg.log;
filter.center = chanlist2idx(cfg.center);
filter.surround = chanlist2idx(cfg.surround);

if isfield (cfg, 'scale_s')
  filter.scale_s = cfg.scale_s;
else
  filter.scale_s = 1;
end

kernFactoryFunc = [cfg.kernel.type 'Kernel'];
filter.kernel = feval (kernFactoryFunc, cfg.kernel);

smlcfg.log = 0;
filter.sml = SML(smlcfg);

filter.channels = mapChannel (cfg.center);
end

function [idx] = chanlist2idx (chanlist)

N = length(chanlist);
idx = zeros (1, N);
for n = 1:length(chanlist)
  ch = chanlist(n);
  idx(n) = str2chan(ch);
end

end

function [channels] = mapChannel (chanlist)

N = length (chanlist);
channels = zeros (1, N*2);

for n = 1:N
  ch = chanlist(n);
  channels(1, (n*2-1)) = str2chan (ch, 'on');
  channels(1, n*2) = str2chan (ch, 'off');
end

end

function [img] = CSRectFilterImage (this, img)

if isfield (this, 'sml')
  sml_filter = this.sml;
  img = sml_filter.function (sml_filter, img);
end

ft = this.kernel;

mp = ceil(length(ft)/2.0);

% remember the weight of the center for later but set it to 0 for
% the convolution of L,M
wc = ft(mp,mp);
ft(mp,mp) = 0;

surs = sum (ft(:));
fprintf ('Filter: S: %f, C: %f; %f\n', surs, wc, wc/abs (surs));
%ft(mp,mp) = wc;

%img.data is 3x256x256 (c,x,y) [f,c,r] of the image 
input = img.data;

% normalize S to the same std as the surround

if this.scale_s == 1
  stdsr = 0;
  for n=1:length(this.surround)
    chan = this.surround(n);
    stdsr = stdsr + std (input(chan,:));
  end
  stdsr = stdsr / length(this.surround);
  stds = std (input(1,:));

  input(1,:,:) = (stdsr / stds) * input(1,:,:);
  fprintf ('Filter: stdsr : stds-> %f \n', stdsr / stds);
end

%stds = std (input(1,:));
%stdml = (std (input(2,:)) + std (input(3,:))) * 0.5;
%input(1,:,:) = (stdml / stds) * input(1,:,:);

% (f,c,r) -> (r,c,f)
input = permute (input, [3 2 1]);

si = size (input);
si = si(1:2);

sk = size (ft);
sn = si-sk+1;
sb = (si-sn)/2;
sv = [1+sb(1):si(1)-sb(1); 1+sb(2):si(2)-sb(2)]';

surround = zeros (sn(1), sn(2));

for n=1:length(this.surround)
  chan = this.surround(n);
  surround = surround + conv2 (input(:,:,chan), ft, 'valid');
end

surround = abs(surround./length(this.surround));

N = length(this.center);
data = zeros(2*N, sn(1), sn(2));

for n=1:length(this.center)
  chan = this.center(n);
  x = (wc * input(sv(:,1),sv(:,1),chan)) - surround;
  [on, off] = rectifyData (x, this.log);
  data((n*2-1),:,:) = on;
  data(n*2,:,:) = off;
end

%adjust the refcoos
rk = img.refkoos;
rk([1,3]) = rk([1,3]) - sb(1);
rk([2,4]) = rk([2,4]) - sb(1);
img.refkoos = rk;

% (f,r,c) -> (f,c,r) [c <-> x, r <-> y]
img.data = permute (data, [1 3 2]);
img.filtered = 1;

fprintf ('\t stats after filtering: Min: %f, Max: %f,\n\t\t Mean: %f, Std: %f \n', ...
  min (data(:)), max (data(:)), mean (data(:)), std (data(:)));
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


