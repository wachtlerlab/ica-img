function [ idx, srv ] = recstrPone(icaall, A, npatch, doplot)

if ~exist('doplot', 'var'); doplot = 1; end;

x = icaall(npatch, :)';
W = pinv(A);
s = W*x;
[~, idx] = sort(-abs(s));
%[~, idx] = sortAbf(A);
srv = zeros(size(A, 2), 1);

for n=1:length(idx)
    rec = A(:,idx(1:n))*s(idx(1:n));
    %rec = A(:,idx(1:n))*s(idx(1:n));
    %pd = var(rec) / var(x);
   d = norm(x - rec);
   pd = d/norm(x);
   srv(n) =  pd;
end

if doplot
    figure;
    hold on;
    plot(1:294, srv)
    plot([1, 294], [0.05, 0.05], 'r')
    plot([1, 294], [0.95, 0.95], 'r')
    plot([1, 294], [1, 1], 'g')
    xlim([1,294])
end

idx = find(srv < 0.05, 1);

end

