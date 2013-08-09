function [ filter ] = CSRect(cfg)
%CSRECT Center-Surround Filter with optional rectification


if ~isfield(cfg, 'log')
  cfg.log = 0;
end

filter.name = 'CSRect';
filter.function = @CSRectFilterImage;
filter.log = cfg.log;
filter.center = chanlist2idx(cfg.center);
filter.surround = mapSurround(cfg.surround, cfg.center);

if isfield (cfg, 'scale_s')
  filter.scale_s = cfg.scale_s;
else
  filter.scale_s = 1;
end

if isfield (cfg, 'normstd')
  filter.normstd = 1;
else
  filter.normstd = 0;
end

if isfield (cfg, 'rectify')
    filter.rectify = cfg.rectify;
else
    filter.rectify = 1;
end

if isfield(cfg, 'center_img')
    filter.center_img = cfg.center_img;
else
    filter.center_img = 0;
end


if ~isfield(cfg, 'lomode')
  cfg.lomode = 0;
end
filter.lomode = cfg.lomode;

if isfield(cfg, 'rectify_mpoint')
   filter.rmpoint = cfg.rectify_mpoint;
end

kernFactoryFunc = [cfg.kernel.type 'Kernel'];
filter.kernel = feval(kernFactoryFunc, cfg.kernel);

smlcfg.log = 0;
if isfield(cfg, 'drift_corr')
  smlcfg.drift_corr = cfg.drift_corr;
end
if isfield(cfg, 'crop')
  smlcfg.crop = cfg.crop;
end

filter.sml = SML(smlcfg);

filter.channels = mapChannel (cfg.center, filter.rectify);
end

function [surround] = mapSurround(surround, center)

if isstruct(surround)
    names = fieldnames(surround);
    idx = cell(length(names), 1);
    for n = 1:length(names)
        ch = char(names(n));
        val = surround.(ch);
        
        if isstruct(val)
            keys = fieldnames(val);
            weights = cell(3, 1);
            for ki = 1:length(keys)
                surr_ch = char(keys(ki));
                surr_idx = str2chan(surr_ch);
                weights{surr_idx} = val.(surr_ch);
            end
            idx{str2chan(ch)} = weights;
        else
            idx{str2chan(ch)} = chanlist2idx(val);
        end
        
    end
    surround = idx;
else
    S = cell(length(center), 1);
    for n = 1:length(center)
        ch = center(n);
        S{str2chan(ch)} = chanlist2idx(surround);
    end
    surround = S;
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

%  norm every channel's std
if this.normstd == 1
  for n = 1:3
    s = std (input(n,:));
    input(n, :,:) = input(n, :, :)/s;
    fprintf ('Filter: STD: [%d] %f\n', n, s);
  end
end

% normalize S to the same std as the surround

if this.scale_s ~= 0
  stdsr = 0;
  S_surround = this.surround{str2chan('S')};
  [channels, weights] = surroundGetChannel(S_surround);
  
  for n=1:length(channels)
    chan = channels(n);
    stdsr = stdsr + weights(n) * std (input(chan,:));
  end
  stds = std (input(1,:));

  input(1,:,:) = (stdsr / this.scale_s*stds) * input(1,:,:);
  fprintf ('Filter: stdsr : stds-> %f \n', stdsr / stds);
end

if this.center_img == 1
    fprintf('Filter: Centering images\n');
    m = mean(input(:));
  for n = 1:3
    input(n, :, :) = input(n, :, :) - m;
  end
end


if this.log && this.lomode == 3
   input = log(input);
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

% surround = zeros (sn(1), sn(2));
% 
% for n=1:length(this.surround)
%   chan = this.surround(n);
%   surround = surround + conv2 (input(:,:,chan), ft, 'valid');
% end
% 
% surround = abs(surround./length(this.surround));
surround = createSurround(this, input, ft, sn);
N = length(this.center);

if this.rectify
   N = 2*N; 
end

data = zeros(N, sn(1), sn(2));


rectifier = cell(N, 1);

for n=1:length(this.center)
  chan = this.center(n);
  x = (wc * input(sv(:,1),sv(:,1),chan)) - surround{chan};
  
  if this.rectify == 1
     [on, off] = rectifyData (x);
     data((n*2-1),:,:) = on;
     data(n*2,:,:) = off;
  elseif this.rectify == 2
      [on, off] = rectifyDataExp (x, this.rmpoint);
      data((n*2-1),:,:) = on(x);
      data(n*2,:,:) = off(x);
      
      rectifier{n*2-1} = on;
      rectifier{n*2}   = off;
  elseif this.rectify == 3
      [on, off] = rectifyDataExpEq (x, this.rmpoint);
      data((n*2-1),:,:) = on(x);
      data(n*2,:,:) = off(x);
  else
      data(n,:,:) = x;
  end
end


if this.log && this.lomode < 3
    if this.lomode == 1
        offset = this.log * max(data(:));
    end
    
    for n=1:N
        x = data(n, :, :);
        if this.lomode == 0
            offset = this.log * max(x(:));
        end
        x = log(x + offset);
        data(n, :, :) = x;
    end
end

if this.log && this.lomode == 4
   data = log(data); 
end

if this.log && this.lomode == 5
   data = log(1 + data); 
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

function [on, off] = rectifyData(data)

on = data;
off = data;

on(on < 0) = 0;
off(off > 0) = 0;
off = -1 * off;

end


function [on, off] = rectifyDataExp(x, rmpoint)
ma = max(abs(x(:)));
s = log(1/rmpoint)/ma;
on  = @(data) ma/exp(s*ma) * exp(s*data);
off = @(data) ma/exp(s*ma) * exp(s*-data);
end

function [on, off] = rectifyDataExpEq(x, rmpoint)

ma = max(x(:));
s = log(1/rmpoint)/ma;
on  = @(data) ma/exp(s*ma) * exp(s*data);

ma = abs(min(x(:)));
s = log(1/rmpoint)/ma;
off = @(data) ma/exp(s*ma) * exp(s*-data);
end

function [channels, weights] = surroundGetChannel(channels)

if isempty(channels)
    channels = [];
    weights = [];
    return;
end

if iscell(channels)
    % cant use cell2mat since that ignores leading zeros
    new_channels = zeros(length(channels),1);
    for idx=1:length(channels)
        new_channels(idx) = channels{idx};
    end
    channels = new_channels;
    valid_channels = channels ~= 0;
    pos = find(valid_channels);
    weights = channels(pos);
    channels = pos;
else
    nch = length(channels);
    weights = ones(nch, 1) / nch;
end

end

function [surround] = createSurround(this, input, ft, sn)
N = length(this.surround);
surround = cell(N, 1);

for ch = 1:length(this.surround)
    channels = this.surround{ch};
    
    if isempty(channels)
        continue;
    end
    
    [channels, weights] = surroundGetChannel(channels);
    
    S = zeros (sn(1), sn(2)); 
    for n=1:length(channels)
        chan = channels(n);
        S = S + weights(n) * conv2 (input(:,:,chan), ft, 'valid');
    end

    surround{ch} = abs(S);
end

end

