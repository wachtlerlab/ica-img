function [ imageset, filter ] = createImageSet (cfg)

source = cfg.source;
basepath = '~/Coding/ICA/'; %FIXME load from config file

dataType = cfg.type;
fprintf('Data Type: %s', dataType);

if strcmp(cfg.type, 'hs_images')
    db_prefix = 'hyperspectral';
else
    db_prefix = cfg.type;
end

% compat hack
dataDir = fullfile (basepath, 'images', db_prefix, source.database);

if iscell(source.images)
    nimages = length(source.images);
else
    nimages = size(source.images, 1);
end

images = cell(nimages, 1);

if isfield (source, 'type')
    imageType = source.type;
else
    imageType = 'rad';
end

for n=1:nimages
  if iscell(source.images)
      filename = source.images{n};
  else
      filename = source.images(n,:);
  end
  
  images{n} = loadImage (filename, dataDir, imageType);  
end

imageset.database = source.database;

% filtering 
if isfield (cfg, 'filter')
  fcfg = cfg.filter;
  
  filterFunc = [fcfg.class];
  filter = feval (filterFunc, fcfg);
  
  if isfield (filter, 'setup')
    filter = filter.setup (filter, images);
  end
  
  for n=1:nimages
    fprintf ('Filtering [%s]\n', images{n}.filename);
    images{n} = filter.function (filter, images{n});
  end
  
  imageset.channels = uint8(filter.channels);
  imageset.filter = filter;
end

imageset.images = images;
imageset.shape = [size(images{1}.data), length(images)];


%shape is [z, x, y, n] with z begin the channel, x, y the image axis and
%      n the number of images

end

