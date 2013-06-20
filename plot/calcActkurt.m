function [ kurt ] = calcActkurt(Act, method, dim)

if ~exist('method', 'var') || isempty(method); method = 'builtin'; end;
if ~exist('dim', 'var') || isempty(dim); dim = 1; end;

if dim ~= 1 && ~strcmpi(method, 'builtin')
    error('dim argument only supported for builtin');
end

if strcmpi(method, 'builtin')
    kurt = doBuiltin(Act, dim);
elseif strcmpi(method, 'stdkurt')
    kurt = doStdKurt(Act);
elseif strcmpi(method, 'beta') || strcmpi(method, 'expwrkur')
    kurt = doExpwrFitKurt(Act);
end

end

function [kurt] = doStdKurt(Act)

nbf = size(Act.w, 2);
kurt = zeros(nbf, 1);
for n=1:nbf
       kurt(n) = stdkurt(Act.w(:, n)); 
end

end

function [kurt] = doExpwrFitKurt(Act)

nbf = size(Act.w, 2);
kurt = zeros(nbf, 1);
[ ~, idx, ~ ] = sortAbf (Act.Model.A);

for n=1:nbf
    data = (Act.w(:, n));
    [~, ~, beta] = expwrfit(data);
    fprintf('beta: %5.3f  [%5.3f] [%5.3f]\n', beta, Act.Model.beta(idx(n)), Act.Model.beta(idx(n)) - beta);
    kurt(n) = expwrkur(beta);
end

end

function [kurt] = doBuiltin(Act, dim)
kurt = kurtosis(Act.w, 0, dim) - 3;
end