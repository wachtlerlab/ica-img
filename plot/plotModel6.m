function plotModel6 (Model)
%PLOTMODEL6 Summary of this function goes here
%   Detailed explanation goes here

hf = figure('Name', ['Basis Functions: ', Model.id(1:7)], 'Position', [0, 0, 1600, 800]);

[~,M] = size (Model.A);
ha = tight_subplot(18, 17, [.01 .03], [.01 .01]);

for idx=1:M

    bf = analyseModel(Model, idx);
    set (gcf, 'CurrentAxes', ha(idx));
    %title (num2str (idx));
    hold on
    image(bf);
    axis image;
    axis off;
end

figure('Name', ['Chrom of BF: ', Model.id(1:7)], 'Position', [0, 0, 1600, 800]);

set (gcf,'Color',[0.9 0.9 0.9])
ha = tight_subplot(18, 17, [.01 .03], [.01 .01]);
pcs = zeros (2, M);

for idx=1:M

    bf = analyseModel(Model, idx, 1);
    colord = analyseModel(Model, idx);
    
    bf = reshape(bf, 7*7, 2);
    
    [pc,~,latent,~] = princomp(bf);
    pc = pc(:,1) * latent (1);
    pcs(:,idx) = pc;
    
    set (gcf, 'CurrentAxes', ha(idx));
    %title (num2str (idx));
    hold on
    scatter (bf(:,1), bf(:,2), 20, reshape (colord, 7*7, 3), 'filled');
    axis ([-1 1 -1 1]);
    %axis image;
    axis off;
end



figure('Name', ['Directions of Chrom BF: ', Model.id(1:7)], 'Position', [0, 0, 800, 800]);

tt = atan2(pcs(2,:), pcs(1,:));
rr = sqrt (pcs(1,:).^2 + pcs(2,:).^2);

xmax = max (abs (rr(:)));

tt = cat (1, tt, tt + pi); %mirror directions
rr = cat (1, rr, rr);

polar (0, xmax); % Scaling of the polar plot to the maximum value
hold on

polar (cat (1, tt, zeros (1, length (tt))),... %theta
       cat (1, rr, zeros (1, length (rr))),... %rho
       '-k');

     
     
figure('Name', ['Directions (Flat) of Chrom BF: ', Model.id(1:7)], 'Position', [0, 0, 1600, 800]);
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

axes ('Position', [.20 .65 .2 .2], 'Layer','top');
plot (kernel_x, kernel);
title ('kernel');

end

function [out] = do_gauss (x, mu, sigma)

out = 1/(2*pi*sigma^2)*exp(-(x - mu).^2/(2*sigma^2));

end