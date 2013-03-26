function [ A ] = prepareAfromModel(Model, nchan)

ps = Model.ds.patchsize;

A = struct();
A.name    = [Model.cfg.id(1:7) '-' Model.id(1:7)];
A.id      = Model.id;
A.nchan   = nchan;
A.nbf     = size(Model.A, 2);
A.ps      = ps;

A.data    = Model.A;
A.sorted  = sortAbf (Model.A);

end

