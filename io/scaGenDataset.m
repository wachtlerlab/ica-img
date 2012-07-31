function [ ds ] = scaGenDataset(path)

sca = SCA(path);
cfg = sca.readConfig();
imgset = createImageSet(cfg.data);
ds = createDataSet(imgset, cfg, 1);
sca.saveDataSet(ds);
sca.close();

end

