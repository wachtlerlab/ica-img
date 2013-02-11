function [ W ] = whiten_filter (data)
%whiten_filter whiten the input
%   rows of data are observations
%   (c) Christian Kellner <kellner@bio.lmu.de>

X = bsxfun(@minus, data, mean(data, 2));

[~, nelm] = size(data);
C = (X*X')/(nelm-1);
[E, D] = eig(C);

W = E * diag(1./(diag(D)).^(1/2)) * E';

end

