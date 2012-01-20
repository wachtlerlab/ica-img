function [ res ] = analyseModel( Model, idx, doxy)

Model = sortModelA(Model);
A = Model.A;

   % S+  S-  M+  M-  L+  L-
X = [ 0,  0, -1,  1,  1, -1;  % x  
      1, -1,  0,  0,  0,  0;  % y 
      0,  0,  1, -1,  1, -1]; % z 

bf1 = reshape (A(:,idx), 6, 7*7);
%
% L = (x + z) * 0.5           = 0.5 x + 0.5z =  0.5 x + 0   y + 0.5 z
% M = z - L = z - 0.5x - 0.5z = 0.5 z - 0.5x = -0.5 x + 0   y + 0.5 z
% S = y + 0.5 L+M             =     y + 0.5z =  0.0 x + 1   y + 0.5 z

if nargin < 3
  XYZ = [ 0.5,    0, 0.5;   %L
         -0.5,    0, 0.5;   %M
            0,    1, 0.5];  %S 
          
    %L-M     L+M-S   L+M
  D2R = ...
  [ 1.0000,  0.1462, 1.0;
   -0.3900, -0.2094, 1.0;
    0.0180,  1.0000, 1.0];
 
 XYZ=D2R;
 
else     
  XYZ = [ 1, 0, 0; 0, 1, 0];
end

%XYZ = [0, 1, 1; -0.5, -0.5, 1; 0.5, 0, 0.5];

[c,~] = size(XYZ); 

res = XYZ*X*bf1;
res = normA (res);
res = permute (reshape (res, c, 7, 7), [3, 2, 1]);

end

function [normed] = normA (A)
normed = 0.5 + 0.5 * A / max(abs(A(:))); 
end