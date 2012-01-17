function plotModel6d( Model)

hf = figure('Name', ['Basis Functions: ', Model.id(1:7)], 'Position', [0, 0, 1600, 800]);

X = [ 0,  0,  0,  0,  1, -1;    % L
      0,  0,  1, -1,  0,  0;    % M 
      1,  -1, 0,  0,  0,  0];   % S

[~,M] = size (Model.A);
Model = sortModelA(Model);
A = Model.A;
bfs = reshape (A, 6, 7*7*M);
imgx = permute (reshape (X*bfs, 3, 7, 7, M), [3 2 1 4]);

ha = tight_subplot(18, 17, [.01 .03], [.01 .01]);

for idx=1:M
  p = imgx(:,:,:,idx);
  bf = 0.5+0.5*p/max(abs(p(:)));
  set (gcf, 'CurrentAxes', ha(idx));
  %title (num2str (idx));
  hold on
  image(bf);
  axis image;
  axis off;
end

hold off;
hf = figure('Name', ['Chrom: ', Model.id(1:7)], 'Position', [0, 0, 1600, 800]);
set(0,'CurrentFigure', hf);
hb = tight_subplot(18, 17, [.01 .03], [.01 .01]);

pcs = zeros (2, M);
for idx=1:M

    p = imgx(:,:,:,idx);
    bf = 0.5+0.5*p/max(abs(p(:)));
    set (gcf, 'CurrentAxes', hb(idx));
    %title (num2str (idx));
    hold on
    slice = reshape (bf, 49, 3);
    
    L = squeeze (slice(:,1));
    M = squeeze (slice(:,2));
    S = squeeze (slice(:,3));

    x = L-M;
    y = S-((L+M)/2);
    
    [pc,~,latent,~] = princomp([x y]);
    pc = pc(:,1) * latent (1);
    pcs(:,idx) = pc;
    
    hold on
    scatter (x, y, 20, slice, 'filled');
    axis ([-1 1 -1 1])
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
     
%%     
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