function checkConfig(cfg)

if (ischar (cfg))
  cfg = loadConfig (cfg, getDefaults('ica.configdir'));
end

checkGradient (cfg);
checkFilter (cfg);
hf = figure('Name', ['Cfg: ', cfg.id(1:7)], 'Position', [0, 0, 1600, 800]);
imageset = createImageSet (cfg.data);
displayImages (imageset, hf);

end

