function [ m_nbf, nbf ] = recstrPatch(icaall, A)

N = size(icaall, 1);

for n=1:100
    nbf = recstrPone(icaall, A, n, 0);
end

m_nbf = mean(nbf);

end

