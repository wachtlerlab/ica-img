function [ dirs ] = calcAdirs( A )

pcs = A.pcs;

tt = atan2(pcs(2,:), pcs(1,:));
%rr = sqrt (pcs(1,:).^2 + pcs(2,:).^2);
rr = hypot(pcs(1,:), pcs(2,:));

tt(tt < 0) = tt(tt < 0) + pi; % negative values get positive here

tt = cat (2, tt, tt + pi); %mirror directions (0-180 -> 180-360)
rr = cat (2, rr, rr);

dirs = zeros(length(tt), 2);

dirs(:, 1) = tt;
dirs(:, 2) = rr;

end

