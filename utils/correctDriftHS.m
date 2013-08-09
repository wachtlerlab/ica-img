function [ hs_corr, data ] = correctDriftHS(hs_data)

nplanes = size(hs_data, 1);
npix = sqrt(size(hs_data,2 ));


data = reshape(hs_data, nplanes, npix, npix);
hs_corr = zeros(size(data));
shift = zeros (2, 1);
shifts = zeros(2, nplanes - 1);


hs_corr(1, :, :) = data(1, :, :);

for p=1:(nplanes-1)
    target = squeeze(data(p+1,:,:));
    XC = xcorr2(squeeze(data(p,:,:)), target);
    [~, idx] = max(abs(XC(:)));
    [r, c] = ind2sub(size(XC), idx);
    cshift = [r - 256, c - 256];
    
    if (any(abs(cshift) > 20)) 
       fprintf('Skipping huge shift!'); 
       continue;
    end
    
    shifts(:, p) = cshift;
    shift = shift + cshift';
    hs_corr(p+1, :, :) = circshift(target, shift');
    
end
cs_shifts = cumsum(shifts, 2);

start = max(max (cs_shifts, [], 2), 0);
stop = min(min (cs_shifts, [], 2), 0);

hs_corr = hs_corr(:, 1+start(1):end+stop(1), 1+start(2):end+stop(2));
data = data(:, 1+start(1):end+stop(1), 1+start(2):end+stop(2));

max_edge = min (size(hs_corr, 2), size(hs_corr, 3));

hs_corr = reshape(hs_corr(:, 1:max_edge, 1:max_edge), nplanes, max_edge*max_edge);
data = reshape(data(:, 1:max_edge, 1:max_edge), nplanes, max_edge*max_edge);

end

