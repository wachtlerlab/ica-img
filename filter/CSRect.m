function [ filter ] = CSRect(cfg)
%CSRECT Summary of this function goes here
%   Detailed explanation goes here

if ~isfield(cfg, 'log')
  cfg.log = '0';
end

filter.name = 'CSRect';
filter.function = @CSRectFilterImage;
filter.log = str2double(cfg.log);
filter.center = chanlist2idx(cfg.center);
filter.surround = chanlist2idx(cfg.surround);
kernFactoryFunc = [cfg.kernel.type 'Kernel'];
filter.kernel = feval (kernFactoryFunc, cfg.kernel);

smlcfg.log = 0;
filter.sml = SML(smlcfg);

end

function [idx] = chanlist2idx (chanlist)

N = length(chanlist);
idx = zeros (1, N);
for n = 1:length(chanlist)
  ch = chanlist(n);
  idx(n) = str2chan(ch);
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

%SML is 256x256x3 (r,c,f) wysiwyg :-> 3x256x256 (f,c,r)
input = permute (img.SML, [3 2 1]); 

% normalize S to the same std as the surround

stdsr = 0;
for n=1:length(this.surround)
   chan = this.surround(n);
  stdsr = stdsr + std (input(chan,:));
end
stdsr = stdsr / length(this.surround);
stds = std (input(1,:));

input(1,:,:) = (stdsr / stds) * input(1,:,:);

%stds = std (input(1,:));
%stdml = (std (input(2,:)) + std (input(3,:))) * 0.5;
%input(1,:,:) = (stdml / stds) * input(1,:,:);

% (f,c,r) -> (r,c,f)
input = permute (input, [3 2 1]);

shape = size (input);
surround = zeros (shape(1), shape(2));
for n=1:length(this.surround)
  chan = this.surround(n);
  surround = surround + conv2 (input(:,:,chan), ft, 'same');
end

surround = abs(surround./length(this.surround));

N = length(this.center);
data = zeros(2*N, shape(1), shape(2));
for n=1:length(this.center)
  chan = this.center(n);
  x = (wc * input(:,:,chan)) - surround;
  [on, off] = rectifyData (x, this.log);
  data((n*2-1),:,:) = on;
  data(n*2,:,:) = off;
end



% (f,r,c) -> (f,c,r)
img.imgData = permute (data, [1 3 2]);
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

% Not needed due to conv (,, 'same')
%
% function [refkoos] = adjustRefKoos (img)
% s = zeros(2, 3);
% s(2,:) = size (img.SML);
% s(1,:) = size (img.imgData);
% 
% sizeDiff = s(1,:) - s(2,:);
% 
% edgeN = s(2,2);
% 
% if edgeN ~= round (edgeN)
%     warning ('ica:adjust_refkoos', 'Image after filtering not square!');
% end
% 
% Img.edgeN = edgeN;
% 
% deltaM = sizeDiff(2);
% deltaN = sizeDiff(3);
% 
% refkoos(1) = Img.refkoos(1) - deltaM;
% refkoos(2) = Img.refkoos(2) - deltaN;
% 
% end
