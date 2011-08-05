function [R, X] = getRatio (Model, a, b, id, writeFile)

Model = sortModelA (Model);
A = Model.A;
A = normABf (A);

[~,M] = size(A);
X = zeros (M, 2);

patchSize = Model.patchSize;
dataDim = Model.dataDim;

for idx = 1:M
    bf = A (:, idx);
    R = reshape (bf, dataDim, patchSize, patchSize);
    X(idx, 1) = do_corr (R(a,:), R(b,:));
    X(idx, 2) = idx;
end

R = X(:, 1);

%X = sortrows (X);
%X(:,2) = [0 diff(X(:,1))']';

%if writeFile
%  if ischar (writeFile) == 0
%    writeFile = 'w+';
%    writeData (Model, X, id, writeFile);
%  end
%end

end

function [out] = do_corr (a, b)
a = a - mean (a);
b = b - mean (b);
out = sum (a.*b);
end

function writeData (Model, M, infix, writeFile)

    nick = Model.id(1:7);
    path = fullfile ('..', 'results', [Model.name '-' nick '-' infix '.txt']);
    
    fd = fopen(path, writeFile);
    
    [nrows, ~] = size (M);
    
    fprintf (fd, '# %s    \t diff    \t Bf-No\n', infix);
    fprintf (fd, '# ------------------------------------\n');
    
    for n = 1:nrows
        fprintf (fd, '% .5f\t% .5f\t %3d\n', M(n,1), M(n,2), M(n,3));
    end
    
    fclose (fd);

end