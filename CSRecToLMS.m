function [ out ] = CSRecToLMS (Model)

out = Model;

C = num2cell (Model.A, 1);

C = cellfun (@selectOn, C, 'UniformOutput', false);
out.A = cell2mat (C);
out.dataDim = 3;

end

function [out] = selectOn (Abf)

shaped = reshape (Abf, 6, 7*7);
%X = shaped([2,4,6], :); % Son, Mon, Lon
%X = shaped([1,3,5], :); % Soff, Moff, Loff
%X = shaped([2,3,6], :); % Son, Moff, Lon
X = shaped([1,4,5], :); % Son, Moff, Lon

out = reshape (X, 3*7*7, 1);


end