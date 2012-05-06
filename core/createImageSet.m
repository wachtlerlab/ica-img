function [ imageset ] = createImageSet (cfg)

source = cfg.source;

% compat hack
dataDir = fullfile ('..', 'data', source.database);


if isfield (cfg, 'filter')
  fcfg = cfg.filter;
  
  filterFunc = [fcfg.class];
  filter = feval (filterFunc, fcfg);
else
  filter = [];
end

nimages = length(source.images);
images = cell(nimages, 1);

for n=1:nimages
  filename = source.images{n};
  img = loadImage (filename, dataDir);
  
  if isstruct(filter)
    img  = filter.function (filter, img);
  end
  
  images{n} = img;
end

imageset.database = source.database;
imageset.images = images;
imageset.shape = [size(images{1}.data), length(images)];

%shape is [z, x, y, n] with z begin the channel, x, y the image axis and
%      n the number of images

end

