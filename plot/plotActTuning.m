function [ fig ] = plotActTuning(A, bfnr)

fig = figure;
nchan = 3;
ps = 7;

bf = A.lms(:, bfnr);

x = 0:(2*pi/360):2*pi;
Nx = length(x);

gain = 0.5;

L = 1 + gain * cos(x);
M = 1 - gain * cos(x);
S = 1 + gain * sin(x);

lms = reshape(bf, nchan, ps*ps);

T = zeros(nchan,  ps*ps, Nx);
T(1, :, :) = lms(1,:)'*L;
T(2, :, :) = lms(2,:)'*M;
T(3, :, :) = lms(3,:)'*S;

X = squeeze(sum(T, 2));

%dklx = X(1,:)-X(2,:);
%dkly = X(3,:)-((X(1,:)+X(2,:))/2);
%z = L+M;
 
%l = hypot(dklx, dkly);
l = sum(X, 1);
polar(x, l);
hold on;

[val, idx] = max(abs(l));
polar([x(idx) x(idx)], [0, l(idx)]);

title(['Tuning BF: ' num2str(bfnr)])

ax = axes('Units','normalized', ...
    'Position',[0.8 0.8 0.2 0.2], ...
    'XTickLabel','', ...
    'YTickLabel','');

set (gcf, 'CurrentAxes', ax);
bfrgb   = reshape(A.rgb(:,bfnr), A.nchan, ps, ps);
bfrgb   = permute (bfrgb, [3 2 1]);
imagesc (bfrgb);
axis image off;

end

