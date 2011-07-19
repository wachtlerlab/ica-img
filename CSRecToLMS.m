function [ out ] = CSRecToLMS (Model, cols)

out = Model;

if length (cols) == 1
               %  S    M    L 
   m = [2,4,6; % on,  on,  on
        1,3,5; % off, off, off
        2,3,6; % on,  off, on
        1,4,5  % on,  off, on
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