function [ cfg ] = icaConfigLoad(cfg)

if (ischar (cfg))
    basepath = getDefaults('ica.basedir');
    cfgpath = fullfile(basepath, 'config');
    cfg = loadConfig (cfg, cfgpath);
    cfg.basepath = basepath;
    cfg.data.basepath = fullfile(basepath, 'images');
end

end

