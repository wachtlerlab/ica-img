function [ r,p ] = corrBfs( Model, bf )

Model = sortModelA (Model);
[~,M] = size(Model.A);
A = Model.A;
bfs = A(:,bf);

shaped = reshape (bfs, 6, 7*7);
[r,p] = corrcoef(shaped');


% c = combnk(1:6,2)';
% [~, n] = size(c);
% bfc = reshape (shaped(c,:)', 49, 2 , n);
% 
% crs = zeros(n,2);
% for idx = 1:n
%   [r,p] = corrcoef(bfc(:,:,idx));
%   crs(idx,:) = [r(1,2),p(1,2)];
% end
  
end

