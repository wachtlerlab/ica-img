function [ angle, distance ] = calcATuningSingle(A, bfnr )
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

[distance, idx] = max(abs(l));

angle = x(idx);

end

