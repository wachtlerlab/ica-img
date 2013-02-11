function checkImages (cfg)

if ischar(cfg)
  cfg = loadConfig (cfg, getDefaults('ica.configdir'));
end

imgset = createImageSet (cfg.data);

displayImages(imgset);

% figure()
% load ('brcmap');
% for n=1:length(imgset.images)
%   
%   img = imgset.images{n};
%   data = img.data;
%   x = reshape (data, shape(1), shape(2) * shape(3));
%   C = corrcoef (x')
%   subplot(2,length(imgset.images),n)
%   imagesc(C)
%   colormap(cool)
%   
%   C = Whitening(img, 5)
%   subplot(2,length(imgset.images),length(imgset.images)+n)
%   imshow(C);
%   colormap(cool)
% end

end

