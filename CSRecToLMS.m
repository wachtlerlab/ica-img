function [ out ] = CSRecToLMS (Model, cols)
%CSRectTOLMS Convert CSRect data to LMS 
% Supports shortcuts for $cols:
%
%      S    M    L 
% 1 = on,  on,  on
% 2 = off, off, off
% 3 = on,  off, on
% 4 = on,  off, on
%
% Son  = 1
% Soff = 2
% Mon  = 3
% Moff = 4
% Lon  = 5
% Loff = 6

out = Model;

if length (cols) == 1
               %  S    M    L 
   m = [1,3,5; % on,  on,  on
        2,4,6; % off, off, off
        1,4,5; % on,  off, on
        2,3,6; % off, on,  off
        ];
   cols = m(cols, :);
end

C = num2cell (Model.A, 1);
selChan = @(X) selectChannel (X, cols);
C = cellfun (selChan, C, 'UniformOutput', false);
out.A = cell2mat (C);
out.dataDim = 3;

end

function [out] = selectChannel (Abf, cols)

shaped = reshape (Abf, 6, 7*7);
X = shaped(cols, :); 
out = reshape (X, 3*7*7, 1);

end