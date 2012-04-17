function [ data_path ] = savePIC(modelId, Model, dataset, fitPar)
%SAVEPIC Save a Precompiled ICA Configuration (PIC)
%        to a hdf5 file

% hack for now
dataset.Ainit = Model.A;

pic_id = [modelId '_' Model.cfgId(1:7) '.pic'];
data_path = [pic_id '.h5'];

fd = H5F.create (data_path, 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');

gcpl = H5P.create ('H5P_LINK_CREATE');
H5P.set_create_intermediate_group (gcpl, 1);
group = H5G.create (fd, '/ds', gcpl,'H5P_DEFAULT', 'H5P_DEFAULT');
  
writeAttrInt (group, 'patchsize', dataset.patchsize);
writeAttrInt (group, 'npats', dataset.npats);
writeAttrInt (group, 'blocksize', dataset.blocksize);
writeAttrInt (group, 'nclusters', dataset.nclusters);
  
saveData (fd, '/ds/indicies', dataset.indicies);
saveData (fd, '/ds/patsperm', dataset.patsperm);
saveData (fd, '/ds/imgdata', dataset.imgdata);
saveData (fd, '/ds/Ainit', dataset.Ainit);
  
H5G.close (group);

group = H5G.create (fd, '/prior', gcpl, 'H5P_DEFAULT', 'H5P_DEFAULT');
saveData (fd, '/prior/beta', Model.prior.beta);
saveData (fd, '/prior/mu', Model.prior.mu);
saveData (fd, '/prior/sigma', Model.prior.sigma);
H5G.close (group);

group = H5G.create (fd, '/fit', gcpl, 'H5P_DEFAULT', 'H5P_DEFAULT');
epsilon = [fitPar.iterPts; fitPar.epsilon]';
saveData (fd, '/fit/epsilon', epsilon);
H5G.close (group);

H5F.close (fd);

end



function [dtype] = class2h5 (data)

k = class(data);

switch (k)
  case 'uint16'
    dtype = 'H5T_NATIVE_USHORT';
    
  case 'int32'
    dtype = 'H5T_NATIVE_INT';
    
  case 'double'
    dtype = 'H5T_NATIVE_DOUBLE';
    
  otherwise
    error ('Implement me!');
end

end

function [dsid] = saveData (fd, name, data)

dims = fliplr (size (data));
dsp = H5S.create_simple (length(dims), dims, dims);
dtype = H5T.copy(class2h5(data));

dset = H5D.create(fd, name, dtype, dsp, 'H5P_DEFAULT');
H5D.write (dset, 'H5ML_DEFAULT','H5S_ALL','H5S_ALL', ...
  'H5P_DEFAULT', data);

if nargout < 1
  H5D.close(dset);
else
  dsid = dset;
end

H5S.close(dsp);
H5T.close(dtype);

end

function [dsid] = writeAttrInt (loc, name, value)

space = H5S.create('H5S_SCALAR');
attr = H5A.create (loc, name, 'H5T_NATIVE_INT', space, 'H5P_DEFAULT');
H5A.write (attr, 'H5T_NATIVE_INT32', int32 (value(1)));
H5A.close (attr);
H5S.close (space);


end

