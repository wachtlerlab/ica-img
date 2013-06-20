function [cog] = calcAcog( A )
 
X = squeeze(mean(A.dkl, 1));

tt = atan2(X(2,:), X(1,:)); % 1 == x, 2 == y
rr = hypot(X(1,:), X(2,:));

cog = [tt; rr]';

end

