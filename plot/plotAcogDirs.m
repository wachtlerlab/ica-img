function [hf_dir, hf_plane] = plotAcogDirs(A, equal)
if ~exist('equal', 'var'); equal = 0; end; 

tt = A.cog(:, 1)';
rr = A.cog(:, 2)';

name = ['Dirs of COG ' A.name];

hf_dir = figure('Name', [name ' [Polar]'], 'Color', 'w', ...
    'Position', [0, 0, 400, 400]);
set(0,'CurrentFigure', hf_dir);

N = length(tt);

if equal
    rr = ones(1, N)/N;
    color = [63, 197, 38] / 255;
    lw = 0.8;
else
    color = [253, 161, 40]/255;
    lw = 1.1;
end

rmax = 0.7*1.05; %max (abs (rr(:)));

%polar (0, xmax); % Scaling of the polar plot to the maximum value
polar(0, rmax);
hold on

polar (cat (1, tt, zeros (1, length (tt))),... %theta
       cat (1, rr, zeros (1, length (rr))),... %rho
       '-k');

hf_plane = figure('Name', [name ' [Plane]'], 'Color', 'w', ...
    'Position', [0, 0, 400, 200]);
set(0,'CurrentFigure', hf_plane);

tt = mod((tt + pi), 2*pi);
tt = tt * 180 / pi;

x = [tt; tt];
y = [zeros(1,length(rr)); rr];
%y = y / max(y(:));

hold on
line (x, y, 'Color', color, 'LineWidth', lw);
xlim([0 360])
ylim([0 rmax])
xlabel('hue angle (degree)')
ylabel('distance')

scale  = 100;

l = length (tt);
vals = zeros (1,360*scale);
for n = 1:l
  x = abs (ceil (tt(n) * scale)) + 1;
  y = rr(n);
  vals(x) = y;
end


kernel_std = 7;
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
