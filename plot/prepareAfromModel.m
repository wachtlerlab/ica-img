function [ A ] = prepareAfromModel(Model, nchan)

ps = Model.ds.patchsize;

A = struct();
A.name    = [Model.cfg.id(1:7) '-' Model.id(1:7)];
A.id      = Model.id;
A.nchan   = nchan;
A.nbf     = size(Model.A, 2);
A.ps      = ps;

A.data    = Model.A;


[sortedA, idx, nA] = sortAbf(Model.A);

A.sorted  = sortedA;
A.norm    = nA(idx);
A.bfidx   = idx;

if isfield(Model, 'beta')
    A.beta    = Model.beta(idx);
    A.kurt    = expwrkur(A.beta);
end


end

