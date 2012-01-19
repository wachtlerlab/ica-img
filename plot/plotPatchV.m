function [ x ] = plotPatchV( Model, idx, fh)

Model = sortModelA(Model);
A = Model.A;

Lp = [ 1  0  1]';
Lm = [-1  0 -1]';
Mp = [-1  0  1]';
Mm = [ 1  0 -1]';
Sp = [ 0  0  0]';
Sm = [ 0 -1  0]';

XYZ =...
   [1.0000,  1.0000, -0.1462;
    1.0000, -0.3900,  0.2094;
    1.0000,  0.0180, -1.0000];
 
D2R = ...
  [1.0000, -0.1462, 1.0;
   -0.3900,  0.2094, 1.0;
   0.0180, -1.0000, 1.0];
  
bf = A(:,idx);
n = length(bf)/6;
x = zeros(n,3);
for pix=1:n
  a = ((pix-1)*6)+1;
  b = a+5;
  p = bf(a:b);
  sp=p(1);
  sm=p(2);
  mp=p(3);
  mm=p(4);
  lp=p(5);
  lm=p(6);
  v = lp*Lp+lm*Lm+mp*Mp+mm*Mm+sp*Sp+sm*Sm;
  %l = (v(1)+v(3))/2;
  %m = (v(3)-v(1))/2;
  %s = (v(2)+v(3)/2);
  dkl = v ([3 1 2],:);
  rgb = D2R*v;
  x(pix,:) = rgb;%[l m s]';
end

x = 0.5 + 0.5*(x/max(abs(x(:))));
img = reshape(x, 7, 7, 3);
if nargin < 3
  figure();
else
  figure(fh);
end

image(img);
axis image
axis off
end

