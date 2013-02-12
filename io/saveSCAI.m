function [ id ] = saveSCAI (Model)

cfg = Model.cfg;
dataset = Model.ds;


cfgid = cfg.id;
id = cfgid(1:7);

filename = [id '.sca.h5'];

fd = H5File(filename);

cfg = rmfield (cfg, 'id');
ctime = cfg.ctime;
cfg = rmfield (cfg, 'ctime');

root = ['/ICA/' id];
group = fd.createGroup(root);
group.set('id', cfgid);
group.set('ctime', ctime);
group.close()

txt = savejson('', cfg);
config = regexprep(txt, '\t', '    ');

ds = fd.write([root '/config'], config);
ds.set ('creator', cfg.creator);
ds.close();

if nargin > 1 && ~isempty(dataset)
  saveDataSet (fd, root, dataset);
end

if nargin > 2 && ~isempty(Model)
  saveModel(fd, root, Model)
end

fd.close();

end

function saveDataSet(fd, root, dataset)

ds_id = dataset.id;
loc = [root '/dataset/' ds_id(1:7)];

group = fd.createGroup(loc);
group.set ('id', ds_id);
group.set ('cfg', dataset.cfg);
group.set ('creator', dataset.creator);
group.set ('dim', int32(dataset.dim));
group.set ('patchsize', int32(dataset.patchsize));
group.set ('channels', dataset.channels);
group.set ('npats', int32(dataset.npats));
group.set ('blocksize', int32(dataset.blocksize));
group.set ('nclusters', int32(dataset.nclusters));
group.set ('maxiter', int32(dataset.maxiter));
group.set ('rng', dataset.rng);

group.close();

fd.write ([loc '/indicies'], dataset.indicies);
fd.write ([loc '/patsperm'], dataset.patsperm);
fd.write ([loc '/imgdata'], dataset.imgdata);
fd.write ([loc '/A_guess'], dataset.Aguess);

end

function saveModel(fd, root, Model)

loc = [root '/model/' Model.id(1:7)];

group = fd.createGroup(loc);
group.set ('id', Model.id);
group.set ('cfg', Model.cfg);
group.set ('ds', Model.ds);
group.set ('creator', Model.creator);
group.set ('ctime', Model.ctime);

if isfield (Model, 'algorithm')
    group.set ('algorithm', Model.algorithm);
end

if isfield (Model, 'onGPU')
  group.set ('gpu', uint16(Model.onGPU));
end

if isfield (Model, 'fit_time')
  group.set ('fit_time', Model.fit_time);
end

group.close();

ds = fd.write([loc '/A'], Model.A);
ds.close();


if isfield (Model, 'prior')
    fd.write ([loc '/beta'], Model.prior.beta);
end

if isfield (Model, 'logE')
    fd.write ([loc '/logE'], Model.logE);
end

end
