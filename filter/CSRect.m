function [ filter ] = CSRect( )
%CSRECT Summary of this function goes here
%   Detailed explanation goes here

filter.name = 'CSRect';
filter.function = @CSRectFilterImage;
filter.kernel = mhat(7, 0.5, 3, 1.785, 0, 0);
filter.log = 0;
filter.center = [1, 2, 3];
filter.surround = [2, 3];

end


function [out] = CSRectFilterImage (this, img)

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
out = permute (data, [1 3 2]);

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

