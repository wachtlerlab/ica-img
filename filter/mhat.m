function [ Z ] = mhat (nvals, si, so, r, mu, debug)

if nargin<4
  mu=0;
end

n = floor (nvals/2.0)

[X, Y] = meshgrid(-n:n);
[~, R] = cart2pol (X, Y);
Z = dog (R, mu, si, so, r);
Z = Z/max(abs(Z(:)));

if nargin > 4 && debug > 0
  n = floor (length(X)/2)+1;
  figure ('Name', 'DoG', 'Position', [0, 0, 600, 1200]);
  subplot(2,1,1);
  hold on;
  plot(X(n,:), Z(n,:));
  plot(X(n,:), Z(n,:), 'r+');
  subplot(2,1,2);
  mesh (X, Y, Z);
  center = Z(n,n);
  Z(n,n) = 0;
  fprintf ('sum: %f; ratio: %f', sum(Z(:)), center/sum(Z(:)));
  Z(n,n) = center;
end



end

