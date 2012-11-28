function [filter] = checkFilter (cfg)

if (ischar (cfg))
  cfg = loadConfig (cfg);
end


fcfg = cfg.data.filter;
fprintf ('Checking filter for [%s] -> %s\n', cfg.id, fcfg.class);
  
filterFunc = [fcfg.class];
filter = feval (filterFunc, fcfg);

if isfield (filter, 'kernel')
  disp (filter.kernel);
  plotFilterKernel (filter.kernel);
end

end