function [A] = convertAtoLMS(Model, W)

A = prepareAfromModel(Model, 3);
data = A.sorted;

[L,M] = size (data);

ps  = A.ps;
ps2 = ps^2;

nchan = L/ps2;

bfs = reshape (data, nchan, ps2*M);

if nchan == 6
  X = [ 0,  0,  0,  0,  1, -1;    % L
        0,  0,  1, -1,  0,  0;    % M 
        1,  -1, 0,  0,  0,  0];   % S
    
  Y = X*bfs;
  rgb = Y;
elseif nchan == 4
  D = [ 0,  0,  0,  0;    % L
        0,  0,  1, -1;    % M 
        1,  -1, 0,  0];   % S
    
  Y = D*bfs;
  rgb = Y;
elseif nchan == 3
  bfs = reshape (data, 3, ps2*M);
  Y = flipdim(bfs, 1); % SML -> LMS
  rgb = Y;
end

dewhiten = exist('W', 'var') && ~isempty(W);
if dewhiten
    Wi = flipdim(flipdim(pinv(W), 2), 1);
    Ys = reshape (permute(reshape(Y, 3, ps2, M), [2 1 3]), 3*ps2, M);
    Z = Wi*Ys;
    rgb = reshape(permute(reshape(Z, ps2, 3, M), [2 1 3]), 3*ps2, M);
    
%    Wi  = pinv(W);
%    Ys  = reshape (permute(flipdim(reshape(Y, 3, ps2, M), 1), [2 1 3]), 3*ps2, M);
%    Z   = Wi*Ys;
%    rgb = reshape(flipdim(permute(reshape(Z, ps2, 3, M), [2 1 3]), 1), 3*ps2, M);
end


A.lms     = reshape (Y, 3*ps2, M);
A.rgb_raw = reshape (rgb, 3*ps2, M);
A.rgb = zeros(size(A.rgb_raw));

for idx=1:M
      p = A.rgb_raw(:,idx);
      A.rgb(:,idx) = 0.5+0.5*p/max(abs(p(:)));
end

A.dkl = calcAdkl(A);
A.pcs = calcApcs(A);

end