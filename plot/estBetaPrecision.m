function [ kltable, betakurt  ] = estBetaPrecision( Act )
nbf = size(Act.w, 2);
npats = size(Act.w, 1);
 A = AfromModel(Act.Model);
 kltable = zeros(nbf, 1);
 betakurt = zeros(nbf, 2);
 N = 1000;
 data = zeros(N, 2, nbf);
 
for idx=1:nbf
   mu = mean(Act.w(:,idx));
   sigma = std(Act.w(:,idx));
   [n, c] = hist(Act.w(:,idx), N);
   pe = expwrpdf(c, mu, sigma, A.beta(idx));
   pe = pe/sum(pe) + eps;
   pest = (n/npats) + eps;
   data(:, 1, idx) = pe;
   data(:, 2, idx) = pest;
   kltable(idx) = kldiv(c, pest, pe);
   betakurt(idx, 1) = A.beta(idx);
   betakurt(idx, 2) = kurt(pe);
   
end

[~,idx] = sort(-kltable);

M = 25;
[m, n] = plotCreateGrid(M);

figure;
for k=1:M
   subplot(m, n, k);
   hold on;
   bfidx = idx(k);
   plot(data(:, 1, bfidx), 'k');
   plot(data(:, 2, bfidx), 'r');
   kue = kurtosis(Act.w(:,bfidx));
   kub = expwrkur(A.beta(bfidx));
   fprintf('%d: %5.3f %5.3f\n', bfidx, kue, kub);
end
fprintf('--------\n');
figure;
[~,idx] = sort(kltable);
for k=1:M
   subplot(m, n, k);
   hold on;
   bfidx = idx(k);
   plot(data(:, 1, bfidx), 'k');
   plot(data(:, 2, bfidx), 'r');
   kue = kurtosis(Act.w(:,bfidx));
   kub = expwrkur(A.beta(bfidx));
   fprintf('%d: %5.3f %5.3f\n', bfidx, kue, kub);
end

kmean = mean(kltable);
fprintf('mean kldiv: %5.3f\n', kmean);
end



