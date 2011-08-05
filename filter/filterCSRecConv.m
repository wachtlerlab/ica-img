function [ dataFiltered ] = filterCSRecConv (Img, dataPar)

ft = dataPar.filter;

sur = ft;
sur(2,2) = 0;
surs = sum (sur(:));
fprintf ('Filter: S: %f, C: %f; %f\n', surs, ft(2,2), ft(2,2)/abs (surs));

% remember the weight of the center for later but set it to 0 for
% the convolution of L,M
wc = ft(2,2);
ft(2,2) = 0;

%SML is 256x256x3 (r,c,f) wysiwyg :-> 3x256x256 (f,c,r)

input = permute (Img.SML, [3 2 1]); 

% normalize S to the same std as LM
stds = std (input(1,:));
stdml = (std (input(2,:)) + std (input(3,:))) * 0.5;
input(1,:,:) = (stdml / stds) * input(1,:,:);

% (f,c,r) -> (r,c,f)
input = permute (input, [3 2 1]);

Mc = doConv (input(:,:,2), ft);
Lc = doConv (input(:,:,3), ft);

LM = ((Lc + Mc) * 0.5);


for n=1:3
  % the actual 'filtering'
  x = (wc * input(2:end-1, 2:end-1, n)) - LM;
  [on, off] = rectifyData (x, dataPar.doLog);
  data((n*2-1),:,:) = on;
  data(n*2,:,:) = off;
end

% (f,r,c) -> (f,c,r)
data = permute (data, [1 3 2]);

if isfield(dataPar, 'activeChs')
  activeChs = dataPar.activeChs;
else
  [nChans,~,~] = size (data);
  activeChs = 1:nChans; % Use all available channels
end

% select channels now
dataFiltered = data(activeChs,:,:);

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

function [res] = doConv (data, kernel)

res = conv2 (data, kernel);
res = res(3:end-2, 3:end-2);

end
