function [ cfg ] = icaConfigLoad(cfg)

if (ischar (cfg))
  cfg = loadConfig (cfg, getDefaults('ica.configdir'));
end

end

