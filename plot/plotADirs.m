function [ hf_dir ] = plotADirs( A )

ps = 7;

hf_dir = figure('Name', ['Dirs of Chroma: ', A.name], 'Position', [0, 0, 1200, 400]);
set(0,'CurrentFigure', hf_dir);

pcs = A.pcs;

tt = atan2(pcs(2,:), pcs(1,:));
rr = sqrt (pcs(1,:).^2 + pcs(2,:).^2);

xmax = max (abs (rr(:)));

tt = cat (1, tt, tt + pi); %mirror directions
rr = cat (1, rr, rr);

subplot(1,2,1)
polar (0, xmax); % Scaling of the polar plot to the maximum value
hold on

polar (cat (1, tt, zeros (1, length (tt))),... %theta
       cat (1, rr, zeros (1, length (rr))),... %rho
       '-k');

subplot(1,2,2)
tt = atan2(pcs(2,:), pcs(1,:));
rr = sqrt (pcs(1,:).^2 + pcs(2,:).^2);

tt(tt < 0) = tt(tt < 0) + pi; % negative values get positive here

tt = cat (2, tt, tt + pi); %mirror directions (0-180 -> 180-360)
rr = cat (2, rr, rr);

tt = tt * 180 / pi;

x = [tt; tt];
y = [zeros(1,length(rr)); rr];
y = y / max(y(:));

hold on
line (x, y, 'Color', 'cyan');
xlim([0 180])

scale  = 100;

l = length (tt);
vals = zeros (1,360*scale);
for n = 1:l
  x = abs (ceil (tt(n) * scale)) + 1;
  y = rr(n);
  vals(x) = y;
end

if nargin < 4
  kernel_std = 3;
end
  
kernel_x = -10:1/scale:10;
kernel = do_gauss (kernel_x, 0, kernel_std);
n = length(kernel_x);
cvals = [vals(length(vals)-(n-1):end) vals vals(1:n)];
x_prime = conv (cvals, kernel, 'same');
xpp = x_prime(n+1:end-n);
xpp = (xpp/max(xpp(:)));

plot (0:(1/scale):(360-1/scale),xpp, 'black');

%axes ('Position', [.20 .65 .2 .2], 'Layer','top');
%plot (kernel_x, kernel);
%title ('kernel');

end

function [out] = do_gauss (x, mu, sigma)

out = 1/(2*pi*sigma^2)*exp(-(x - mu).^2/(2*sigma^2));

end
