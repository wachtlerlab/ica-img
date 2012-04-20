function [ images ] = prepareImages (cfg)

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




end

