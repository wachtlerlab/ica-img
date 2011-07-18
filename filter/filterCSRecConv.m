function [ data ] = filterCSRecConv (Img, dataPar)

ft = dataPar.filter;

sur = ft;
sur(2,2) = 0;
%surs = sum (sur(:));
%fprintf ('Filter: S: %f, C: %f; %f\n', surs, ft(2,2), ft(2,2)/abs (surs));

% remember the weight of the center for later but set it to 0 for
% the convolution of L,M
wc = ft(2,2);
ft(2,2) = 0;

input = permute (Img.SML, [3 2 1]);

% normalize S to the same std as LM
stds = std (input(1,:));
stdml = (std (input(2,:)) + std (input(3,:))) * 0.5;
input(1,:,:) = (stdml / stds) * input(1,:,:);

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

data = permute (data, [1 3 2]);
fprintf ('\n\tImg-stats after filtering: Min: %f, Max: \n\t\t %f, Mean: %f, Var: %f \n', ...
  min (data(:)), max (data(:)), mean (data(:)),var (data(:)));
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
  on = log (on + 0.01 * max (on(:)));
  off = log (off + 0.01 * max (off(:)));
end

end

function [res] = doConv (data, kernel)

res = conv2 (data, kernel);
res = res(3:end-2, 3:end-2);

end
