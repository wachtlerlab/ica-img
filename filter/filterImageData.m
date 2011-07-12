function [ ftData ] = filterImageData (imageData, dataPar)

ft = dataPar.filter;

ftData(1,:,:) = conv2 (squeeze(imageData(1,:,:)), ft, 'valid');
ftData(2,:,:) = conv2 (squeeze(imageData(2,:,:)), ft, 'valid');
ftData(3,:,:) = conv2 (squeeze(imageData(3,:,:)), ft, 'valid');

end

