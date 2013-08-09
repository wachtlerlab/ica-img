function [ diffs ] = squaredDiffHS(hs_data)


nplanes = size(hs_data, 1);
npix = size(hs_data, 2);
diffs = zeros(nplanes - 1, 1);


for p=1:(nplanes-1)
    target = squeeze(hs_data(p+1,:));
    diffs(p) = sum((squeeze(hs_data(p,:)) - target).^2) / npix;
end


end