function [ hf_dir ] = plotATuning(A)

nbf = A.nbf;

angles = zeros(nbf, 1);
distances = zeros(nbf, 1);
for bfidx = 1:nbf
    [theta, r] = calcATuningSingle(A, bfidx);
    angles(bfidx, :) = theta;
    distances(bfidx, :) = r;
end

%rr = ones(nbf, 1);

[hf_dir, hf_plane] = plotDirs(['Tunings'], angles', distances', 0);
[kest, ui] = calcKLDiv( angles');

end

