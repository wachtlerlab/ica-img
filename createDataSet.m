function [dataset] = createDataSet (modelId, filename)

% basic init
clear Model fitPar dispPar Result;

stateDir = fullfile ('..', 'state');

if exist (stateDir, 'dir') == 0
   mkdir (stateDir); 
end

if exist (fullfile ('config', [modelId '.m']), 'file') == 0
  error ('Cannot find config');
end

[Model, fitPar, dispPar, dataPar] = loadConfig (modelId);

images = prepareImages (dataPar);

tstart = tic;
fprintf('\nGenerating dataset...\n');
dataset = generateDataSet (images, fitPar, dataPar);
fprintf('   done in %f\n', toc(tstart));

[c, ~, ~] = size (dataset.imgdata);
N = dataset.patchsize^2 * c;
dataset.Ainit = randn (N, N) + 0.5 * eye(N, N);

if nargin > 1

  fd = H5F.create (filename, 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');

  gcpl = H5P.create ('H5P_LINK_CREATE');
  H5P.set_create_intermediate_group (gcpl, 1);
  group = H5G.create (fd, '/ds/0', gcpl,'H5P_DEFAULT', 'H5P_DEFAULT');
  
  writeAttrInt (group, 'patchsize', dataset.patchsize);
  writeAttrInt (group, 'npats', dataset.npats);
  writeAttrInt (group, 'blocksize', dataset.blocksize);
  writeAttrInt (group, 'nclusters', dataset.nclusters);
  
  saveData (fd, '/ds/0/indicies', dataset.indicies);
  saveData (fd, '/ds/0/patsperm', dataset.patsperm);
  saveData (fd, '/ds/0/imgdata', dataset.imgdata);
  saveData (fd, '/ds/0/Ainit', dataset.Ainit);
  
  H5G.close (group);
  H5F.close (fd);
end
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