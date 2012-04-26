function [ model ] = setupModel (cfg, dataset)
%CREATEMODEL Summary of this function goes here

dim = dataset.dim;
cfgid = cfg.id;

prior = createPrior(cfg.prior, dim);

model.ds = dataset.id;
model.cfg = cfgid;
model.creator = genCreatorId();
model.prior = prior;
model.A = dataset.Aguess;

id = DataHash (model, struct ('Method', 'SHA-1'));
model = setfield_idx(model, 'id', id, 1);


end

