function [ sort_idx ] = plotDiCorr(activations)
X = activations;
s = size (activations);
Y = reshape (X, s(1)*s(2), s(3));
cor = corrcoef (Y);

%figure;
%imagesc (cor);

simidx = zeros (size (cor));
[M, N] = size (cor);
for a = 1:M
  for b = a:M
    simidx(a, b) = sim_index(cor, a, b);
    simidx(b, a) = simidx(a, b);
  end
end

scores = zeros(M, 1);
indicies = zeros(M, M);

for k=1:M
  [ix, score] = sort_corr (simidx, k);
  indicies(k, :) = ix';
  scores(k) = score;
end

[~, li] = sort (scores);

for n = 1:1
sort_idx = indicies(li(n),:);
plotResult(cor, sort_idx);
end


figure();
imagesc(simidx);
figure;
%subplot(211);
imagesc(cor);
colormap(hot);
%figure();
%subplot(212);


end


function plotResult(cor, sort_idx)
load ('brcmap')
Scor = cor(sort_idx, :);
Scor = Scor(:, sort_idx);
figure;
imagesc (Scor);
colormap(mycmp2);

end

function [indicies, score] = sort_corr(simidx, start)

[M, N] = size (simidx);
unsorted = ones(M, 1);
indicies = zeros(M, 1);

idx = start;
indicies(1) = idx;
unsorted(idx) = 0;
cs_val = 0;

for n = 2:M
  cur_row = simidx(idx, :);
  [vals, min_idx] = sort (cur_row);
  [idx, unsorted] = find_avail_index(min_idx, unsorted);
  cs_val = cs_val + vals(idx);
  indicies(n) = idx;
end

score = cs_val / (length(indicies)-1);
%disp(cs_val);

end

function [idx, unsorted] = find_avail_index (min_idx, unsorted)
N = length (min_idx);

for n = 1:N
  if unsorted(min_idx(n))
    break;
  end
end

idx = min_idx(n);
unsorted(idx) = 0;
end

function [sidx] = sim_index(A, a, b)
[~, N] = size (A);

%sidx = arrayfun (@(x, y) (x-y)^2, A(a, :), A(b, :));
%sidx = sum (sidx);

sidx = 0;
for n = 1:N
  sidx = sidx + (A(a, n) - A(b, n))^2;
end


end