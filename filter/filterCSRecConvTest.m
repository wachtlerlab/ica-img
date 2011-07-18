function [ data ] = filterCSRecConvTest (Img, doLog)

if nargin < 2
  doLog = 1;
end

X = ones (3, 3) * 1.94;
X(2,2) = 0;

%m_size = 3;
%[xx, yy] = meshgrid ((1:m_size) - round(m_size/2));
%[THETA, X] = cart2pol (xx, yy);

%so = 1.3383326406221043;
%si = 0.7477302449634321;
%sc = 3.812773094633841;

%ft = sc * dog (X, 0, si, so)
%ft = mexican_hat (3, 20, 1, 6);
ft = ones (3, 3) * 0.1250;
ft(2,2) = 1.0;


sur = ft;
sur(2,2) = 0;
surs = sum (sur(:));
fprintf ('S: %f, C: %f; %f\n', surs, ft(2,2), ft(2,2)/abs (surs));

wc = ft(2,2);
ft(2,2) = 0;

%S = squeeze (Img.imgData(1,:,:));
%M = squeeze (Img.imgData(2,:,:));
%L = squeeze (Img.imgData(3,:,:));

S = squeeze (Img.SML(:,:,1));
M = squeeze (Img.SML(:,:,2));
L = squeeze (Img.SML(:,:,3));

stds = std (S(:))
stdml = (std (M(:)) + std (L(:))) * 0.5

S = (stdml / stds) * S;


%Sc = conv2 (S, ft);
Mc = conv2 (M, ft);
Lc = conv2 (L, ft);
Mc = Mc(3:end-2, 3:end-2);
Lc = Lc(3:end-2, 3:end-2);
figure('Name', 'Filter');

ha = tight_subplot (5, 3, .01, .09);

colormap ('gray')

np = doPlot (S, 1, ha, 'S');
np = doPlot (L, np, ha, 'M');
np = doPlot (M, np, ha, 'L');

LM = ((Lc + Mc) * 0.5);

Sx = (wc * S(2:end-1, 2:end-1)) - LM;
Mx = (wc * M(2:end-1, 2:end-1)) - LM;
Lx = (wc * L(2:end-1, 2:end-1)) - LM;


%Sx = log (Sx + 0.01 * max(Sx(:)));
%Mx = log (Mx + 0.01 * max(Mx(:)));
%Lx = log (Lx + 0.01 * max(Lx(:)));

np = doPlot (Sx, np, ha, 'S (-LM)');
np = doPlot (Mx, np, ha, 'M (-LM)');
np = doPlot (Lx, np, ha, 'L (-LM)');

Son = Sx;
Soff = Sx;
Mon = Mx;
Moff = Mx;
Lon = Lx;
Loff = Lx;

Son(Son < 0) = 0;
Mon(Mon < 0) = 0;
Lon(Lon < 0) = 0;

if doLog
  Son = log (Son);
  Mon = log (Mon);
  Lon = log (Lon);
end

np = doPlot (Son, np, ha, 'S (-LM) on');
np = doPlot (Mon, np, ha, 'M (-LM) on');
np = doPlot (Lon, np, ha, 'L (-LM) on');

Soff(Soff > 0) = 0;
Moff(Moff > 0) = 0;
Loff(Loff > 0) = 0;

if doLog
  Soff = -1 * log (-1 * Soff);
  Moff = -1 * log (-1 * Moff);
  Loff = -1 * log (-1 * Loff);
end

np = doPlot (Soff, np, ha, 'S (-LM) off');
np = doPlot (Moff, np, ha, 'M (-LM) off');
np = doPlot (Loff, np, ha, 'L (-LM) off');

fprintf ('Min: %f, Max: %f, Mean: %f, Var: %f \n', ...
  min (Sx(:)), max (Sx(:)), mean (Sx(:)),var (Sx(:)));
fprintf ('Min: %f, Max: %f, Mean: %f, Var: %f\n', ...
  min (Mx(:)), max (Mx(:)), mean (Mx(:)), var (Mx(:)));
fprintf ('Min: %f, Max: %f, Mean: %f, Var: %f\n', ...
  min (Lx(:)), max (Lx(:)), mean (Lx(:)), var (Lx(:)));

np = doPlot (LM, np, ha, '(L+M)/2');
np = doPlot (ft, np, ha);
colorbar();
ft(2,2) = wc;
doPlot (ft, np, ha);
colorbar();

%x = -5:0.01:5;
%set (gcf, 'CurrentAxes', ha(12));
%hold on
%plot (x, dog (x, 0, si, so));

end

function [np] = doPlot (data, fidx, fh, txt)

set (gcf, 'CurrentAxes', fh(fidx));
hold on
axis off;
axis equal;
axis ij;
imagesc (data);

if nargin > 3
  text (6, 12, txt, 'color', 'r');
end

np = fidx + 1;

end
