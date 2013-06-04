function [ model ] = fasticaAssembleModel (cfg, dataset, A)

%FIXME: c&p from setupModel, need to unite those two
cfgid = cfg.id;

model.ds = dataset.id;
model.cfg = cfgid;
model.creator = [genCreatorId() ' [fastica]'];
model.ctime = gen_ctime();

id = DataHash (model, struct ('Method', 'SHA-1'));

model.ds = dataset;
model.cfg = cfg;

model.A = A;
model.algorithm = 'fastica';

model = setfield_idx(model, 'id', id, 1);
model.fit_time = 0;
model.onGPU = 0;

end

