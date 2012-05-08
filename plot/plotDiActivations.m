function [ activation ] = plotDiActivations( A, idx )

A = sortAbf(A);

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

if nargin > 1
  [~, sidx] = sort (idx);
end

A = normA(A);
for bfidx = 1:L
  
  if nargin > 1
    cbf = sidx(bfidx);
  else
    cbf = bfidx;
  end
  
  bf = A(:,cbf);
  for m = 1:res
    for n = 1:res
      patch = genPatch(S(m, n), X(m, n), 49);
      activation(m, n, bfidx) = dot (bf, patch);
    end
  end
end

%activation = 0.5 + 0.5*activation/(max(abs(activation(:))));
activation = cell2mat (cellfun (@(x) x/max(abs(x(:))), num2cell (activation,[1:2]), 'UniformOutput', false));

figure()
for bfidx=1:L
  subplot (14, 14, bfidx);
  imagesc (activation(:,:,bfidx), [-1 1]);
  axis off;
end

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

function [patch] = genPatch (S, X, N)

strip = zeros(4,1);
strip(2 - (S > 0)) = S;
strip(4 - (X > 0)) = X;
patch = repmat (strip, [N, 1]);

end

function [X] = normA(A)
X = cell2mat (cellfun (@(x) x/max(abs(x(:))), num2cell (A,1), 'UniformOutput', false));
end

