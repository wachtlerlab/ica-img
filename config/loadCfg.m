function [ cfg ] = loadCfg (configId)

configFile = fullfile ('config', [configId '.json']);

if exist (configFile, 'file') == 0
  error ('Cannot find config');
end

cfg = loadjson(configFile);

hashOpts.Method = 'SHA-1';
hashOpts.isFile = 'true';

cfg_hash= DataHash (configFile, hashOpts);
cfg.id = cfg_hash

end

