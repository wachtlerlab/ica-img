function checkFilter (cfg)

fprintf ('Checking filter for [%s]\n', cfg.id);

fcfg = cfg.data.filter;
  
filterFunc = [fcfg.class];
filter = feval (filterFunc, fcfg);
disp (filter.kernel);
plotFilterKernel (filter.kernel);

end