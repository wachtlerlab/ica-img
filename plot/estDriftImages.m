function [ max_shifts ] = estDriftImages(imageset, step)
nimages = length(imageset.images);

planes = 1:step:31;
nshifts = length(planes) - 1;

max_shifts = zeros(2, nshifts, nimages);

for k=1:nimages
   img = imageset.images{k};
   %xdata = gpuArray(img.hs_data);
   
   xdata = img.hs_data;
   [~, MN] = size(img.hs_data);
   M = sqrt(MN);
   
   data = reshape(xdata, 31, M, M);
  
   for p = planes(1:end-1)
       XC = xcorr2(squeeze(data(p,:,:)), squeeze(data(p+1,:,:)));
       [~, idx] = max(abs(XC(:)));
       [i, j] = ind2sub(size(XC), idx);
       shift = [i - M, j - M];
       max_shifts(:, p, k) = shift;
   end
   fprintf('image: %d \n', k);
end

end

