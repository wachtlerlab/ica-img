function [ activation ] = plotDiActivations( A, idx )
 
A = sortAbf(A);
 
if nargin > 1
  A = A(:, idx);
else
 
end

res = 21;
[phi, r] = meshgrid(linspace(0, 2*pi, res), linspace (-1.0, 1.0, res));

S = sin(phi);
X = (sin(phi - (pi/2)) + r);

%color = lines(res);
%figure ();
%hold on;
%for n = 1:res
%  plot (linspace(0, 2*pi, res), S(n,:), '--','Color', color(n,:))
%  plot (linspace(0, 2*pi, res), M(n,:), '-', 'Color', color(n,:))
%end

debug = 0;
if debug
  figure();
  subplot(211)
  mesh (phi, r, S)
  xlabel('Phi')
  ylabel('R')
  zlabel('S')
  
  subplot(212)
  mesh (phi, r, X)
  xlabel('Phi')
  ylabel('R')
  zlabel('M')
end

L = length(A);

activation = zeros (res, res, L);

A = normA(A);
for bfidx = 1:L
  bf = A(:,bfidx);
  for m = 1:res
    for n = 1:res
      patch = genPatch(S(m, n), X(m, n), 49);
      activation(m, n, bfidx) = dot (bf, patch);
    end
  end
end

%activation = 0.5 + 0.5*activation/(max(abs(activation(:))));
activation = cell2mat (cellfun (@(x) x/max(abs(x(:))), num2cell (activation,[1:2]), 'UniformOutput', false));

%if nargin > 1
%  activation = activation(:, :, idx);
%end

fig = figure();
for bfidx=1:L
  subplot (14, 14, bfidx);
  imagesc (activation(:,:,bfidx), [-1 1]);
  axis off;
end

load ('brcmap.mat')
set(fig, 'Colormap', mycmp2)

if debug
figure;
imagesc (std(activation, [], 3))
title('Std')
xlabel('Phi')
ylabel('R')


figure;
imagesc (max(activation, [], 3))
title('Max')
xlabel('Phi')
ylabel('R')


figure;
imagesc (min(activation, [], 3))
title('Min')
xlabel('Phi')
ylabel('R')
end

end

function [x] = rectify(x, on)
if x < 0 && ~on
  x = abs(x);
elseif x > 0 && ~on
    x = -x;
end

end

function [patch] = genPatch (S, X, N)

strip = zeros(4,1);
%strip(1) = rectify (S, 1);
%strip(2) = rectify (S, 0);
%strip(3) = rectify (X, 1);
%strip(4) = rectify (X, 0);
strip(2 - (S > 0)) = abs(S);
strip(4 - (X > 0)) = abs(X);
patch = repmat (strip, [N, 1]);

end

function [X] = normA(A)
X = cell2mat (cellfun (@(x) x/max(abs(x(:))), num2cell (A,1), 'UniformOutput', false));
end

