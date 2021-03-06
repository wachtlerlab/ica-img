function plotModel6d (Model, do_print)

hf_bf = figure('Name', ['Basis Functions: ', Model.id(1:7)], 'Position', [0, 0, 1600, 800]);

X = [ 0,  0,  0,  0,  1, -1;    % L
      0,  0,  1, -1,  0,  0;    % M 
      1,  -1, 0,  0,  0,  0];   % S

D = [ 0,  0,  0,  0;    % L
      0,  0,  1, -1;    % M 
      1,  -1, 0,  0];   % S

LMS = [   0,   -1,  1;  % X (L-M)
       1, -0.5,  -0.5;  % Y (S - L+M)
          0,    1,  1]; % Z (L+M)
      
      %   X         Y      Z    
D2R = [ 1.0000,  0.1462, 1.0;    % R
       -0.3900, -0.2094, 1.0;    % G
        0.0180,  1.0000, 1.0;];  % B
        
[L,M] = size (Model.A);
Model = sortModelA(Model);
A = Model.A;

nchan = L/49;

bfs = reshape (A, nchan, 7*7*M);

if nchan == 6
  Y = X*bfs;
  rgb = Y;
elseif nchan == 4
  Y = D*bfs;
  %dkl = LMS*Y;
  %rgb = D2R*dkl;
  rgb = Y;
elseif nchan == 3
  bfs = reshape (A, 3, 7*7*M);
  Y = flipdim(bfs, 1); % SML -> LMS
  rgb = Y;
end

imgx = permute (reshape (Y, 3, 7, 7, M), [3 2 1 4]);
img_rgb = permute (reshape (rgb, 3, 7, 7, M), [3 2 1 4]);

root = sqrt (M);
nrows = ceil(root);
ncols = floor(root);
ha = tight_subplot(nrows, ncols, [.01 .03], [.01 .01]);

for idx=1:M
  p = img_rgb(:,:,:,idx);
  bf = 0.5+0.5*p/max(abs(p(:)));
  set (gcf, 'CurrentAxes', ha(idx));
  %title (num2str (idx));
  hold on
  image(bf);
  axis image;
  axis off;
end

hold off;
hf_chrom = figure('Name', ['Chrom: ', Model.id(1:7)], 'Position', [0, 0, 1600, 800]);
set(0,'CurrentFigure', hf_chrom);
hb = tight_subplot(nrows, ncols, [.01 .03], [.01 .01]);

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
    %axis equal;
    %axis image;
end


hf_dir = figure('Name', ['Directions of Chrom BF: ', Model.id(1:7)], 'Position', [0, 0, 800, 800]);

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
hf_dir_flat = figure('Name', ['Directions (Flat) of Chrom BF: ', Model.id(1:7)], 'Position', [0, 0, 1600, 800]);
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

if nargin > 1 && do_print
  nick = [Model.cfg.id(1:7) '-' Model.id(1:7)];

  reportDir = fullfile ('..', 'reports');

  if exist (reportDir, 'dir') == 0
    mkdir (reportDir);
  end

  filePath = fullfile (reportDir, [nick '.ps']);

  do_print_fig (hf_bf, filePath);
  do_print_fig (hf_chrom, filePath);
  do_print_fig (hf_dir, filePath);
  do_print_fig (hf_dir_flat, filePath);


end

end

function do_print_fig (hf, filePath, res, renderer)

if nargin < 3
  res = '300';
end

if nargin < 4
  re = [];
else
  re = ['-' renderer];
end

  print (hf, '-dpsc2', '-append', ['-r' res], re, filePath);

end

function [out] = do_gauss (x, mu, sigma)

out = 1/(2*pi*sigma^2)*exp(-(x - mu).^2/(2*sigma^2));

end
