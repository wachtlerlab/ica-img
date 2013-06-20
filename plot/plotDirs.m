function [ hf_dir, hf_plane ] = plotDirs(name, tt, rr, equal, cutoff)

if ~exist('equal', 'var'); equal = 0; end; 
if ~exist('cutoff', 'var'); cutoff = 0; end; 

hf_dir = figure('Name', [name ' [Polar]'], 'Color', 'w', ...
    'Position', [0, 0, 400, 400]);
set(0,'CurrentFigure', hf_dir);

N = length(tt);

if equal
    rr = ones(1, N)/N;
    color = [252, 61, 54] / 255;
    lw = 0.8;
    ymax = max (abs (rr(:)));
else
    color = [0.121569 0.427451 0.658824];
    lw = 1.1;
    ymax = max (abs (rr(:))); %0.1; %max (abs (rr(:)));
   
end

polar (0, ymax); % Scaling of the polar plot to the maximum value
hold on
    
polar (cat (1, tt, zeros (1, length (tt))),... %theta
       cat (1, rr, zeros (1, length (rr))),... %rho
       '-k');

hf_plane = figure('Name', [name ' [Plane]'], 'Color', 'w', ...
    'Position', [0, 0, 400, 200]);
set(0,'CurrentFigure', hf_plane);

tt = tt * 180 / pi;

tt = mod(tt + 360, 360);

if 0
    idx = find(rr > 0.00125);    
    tt = tt(idx);
    rr = rr(idx);
end

x = [tt; tt];
y = [zeros(1,length(rr)); rr];
%y = y / max(y(:));

hold on
line (x, y, 'Color', color, 'LineWidth', lw);

if cutoff
    xlim([0 180])
else
    xlim([0 360])
end


if ~equal
    ylim([0 ymax])
end

xlabel('hue angle (degree)')
ylabel('eigenvalue')

scale  = 100;

l = length (tt);
vals = zeros (1,360*scale);
for n = 1:l
  x = abs (ceil (tt(n) * scale)) + 1;
  y = rr(n);
  vals(x) = y;
end


kernel_std = 5;
kernel_x = -10:1/scale:10;
kernel = do_gauss (kernel_x, 0, kernel_std)*10;
n = length(kernel_x);
cvals = [vals(length(vals)-(n-1):end) vals vals(1:n)];
x_prime = conv (cvals, kernel, 'same');
xpp = x_prime(n+1:end-n);
%xpp = (xpp/max(xpp(:)));

plot (0:(1/scale):(360-1/scale), xpp, 'black');

%axes ('Position', [.20 .65 .2 .2], 'Layer','top');
%plot (kernel_x, kernel);
%title ('kernel');

end

function [out] = do_gauss (x, mu, sigma)

out = 1/(2*pi*sigma^2)*exp(-(x - mu).^2/(2*sigma^2));

end
