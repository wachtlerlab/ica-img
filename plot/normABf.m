function [out] = normABf (A)

C = num2cell (A, 1);
C = cellfun (@normA, C, 'UniformOutput', false);
out = cell2mat (C);

end


function [normed] = normA (A)
normed = 0.5 + 0.5 * A / max(abs(A(:))); 
end