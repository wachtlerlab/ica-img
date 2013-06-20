function [ hf_dir, hf_plane ] = plotADirs(A, equal)

if ~exist('equal', 'var'); equal = 0; end; 

tt = A.dirs(:, 1)';
rr = A.dirs(:, 2)';

[hf_dir, hf_plane] = plotDirs(['Dirs of Chroma ' A.name], tt, rr, equal);
 
end
