function [ imageset ] = createImageSet (cfg)

source = cfg.source;

% compat hack
dataDir = fullfile ('..', 'images', 'hyperspectral', source.database);

nimages = length(source.images);
images = cell(nimages, 1);

for n=1:nimages
  filename = source.images{n};
  images{n} = loadImage (filename, dataDir);  
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
end

imageset.images = images;
imageset.shape = [size(images{1}.data), length(images)];

%shape is [z, x, y, n] with z begin the channel, x, y the image axis and
%      n the number of images

end

