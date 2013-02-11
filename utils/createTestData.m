function [res] = createTestData (filename)

filename = [filename '.h5'];

fd = H5F.create (filename, 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');

iterPts        =     [  1,    1000,   5000,  10000 30000 60000  90000];
epsSteps       = 20 *[ 0.02,  0.01,  0.005,  0.001 0.0005 0.0001 0.00005];

N = 100;
data = zeros (N, 2);

for n = 1:N
  mi = max(iterPts);
  i = randi ([1, mi]);
  data(n, 1) = i;
  data(n, 2) = interpIter (i, iterPts, epsSteps);
end

%% ---- mean, std ----
gcpl = H5P.create ('H5P_LINK_CREATE');
H5P.set_create_intermediate_group (gcpl, 1);
gt = '/test/math';
group = H5G.create (fd, gt, gcpl,'H5P_DEFAULT', 'H5P_DEFAULT');

M = 128;
X = pi + exp(1).*randn (M,M);
m = mean (X(:));
v = var(X(:));
s = std (X(:));
Xzm = X-m;

saveData (fd, '/test/math/data', X);
saveData (fd, '/test/math/mean', m);
saveData (fd, '/test/math/var', v);
saveData (fd, '/test/math/std', s);
saveData (fd, '/test/math/data_zm', Xzm);


H5G.close (group);

%% ---- interpolate test ----
gcpl = H5P.create ('H5P_LINK_CREATE');
H5P.set_create_intermediate_group (gcpl, 1);
gt = '/test/interpolate';
group = H5G.create (fd, gt, gcpl,'H5P_DEFAULT', 'H5P_DEFAULT');

saveData (fd, [gt '/samples'], data);
epsilon = [iterPts; epsSteps]';
saveData (fd, [gt '/epsilon'], epsilon);

H5G.close (group);
%% ---- pinv ----

A = randn(400,400);
Ai = pinv(A);

gt = '/test/pinv';
group = H5G.create (fd, gt, gcpl,'H5P_DEFAULT', 'H5P_DEFAULT');
saveData (fd, [gt '/A'], A);
saveData (fd, [gt '/Ai'], Ai);

H5G.close (group);

%% ---- dataset ---
[Model, fitPar, dataPar] = loadConfig ('tri_csr_0', getDefaults('ica.configdir'));


%% Prepare image data
images = prepareImages (dataPar);

tstart = tic;
fprintf('\nGenerating dataset...\n');
ds = createDataSet (images, fitPar, dataPar);
fprintf('   done in %f\n', toc(tstart));

ds.Ainit = Model.A;

gcpl = H5P.create ('H5P_LINK_CREATE');
H5P.set_create_intermediate_group (gcpl, 1);
group = H5G.create (fd, '/test/dataset', gcpl,'H5P_DEFAULT', 'H5P_DEFAULT');

writeAttrInt (group, 'patchsize', ds.patchsize);
writeAttrInt (group, 'npats', ds.npats);
writeAttrInt (group, 'blocksize', ds.blocksize);
writeAttrInt (group, 'nclusters', ds.nclusters);


saveData (fd, '/test/dataset/indicies', ds.indicies);
saveData (fd, '/test/dataset/patsperm', ds.patsperm);
saveData (fd, '/test/dataset/imgdata', ds.imgdata);
saveData (fd, '/test/dataset/Ainit', ds.Ainit);

c0 = extractPatches(ds, 1);
saveData (fd, '/test/dataset/c0', c0);

D = c0(:,1:(ds.blocksize-1));
saveData (fd, '/test/dataset/D', D);

sigma = std(D(:));
saveData (fd, '/test/dataset/D_sigma', sigma)

A = ds.Ainit;
[~,M] = size(A);
for m=1:M
  A(:,m) = A(:,m) * sigma;
end

saveData (fd, '/test/dataset/A_scaled', A)

H5G.close (group);

%% ----
group = H5G.create (fd, '/test/ica', gcpl,'H5P_DEFAULT', 'H5P_DEFAULT');
saveData (fd, '/test/ica/A', A);

S = pinv(A)*D;
saveData (fd, '/test/ica/S', S);

[~, M] = size (A);
ExPwr.mu     = zeros(M,1);
ExPwr.sigma  = ones(M,1);
ExPwr.beta   = 0.5*ones(M,1);
ExPwr.a      = 2;
ExPwr.b      = 2;
ExPwr.tol    = 0.1;

saveData (fd, '/test/ica/mu', ExPwr.mu);
saveData (fd, '/test/ica/beta', ExPwr.beta);
saveData (fd, '/test/ica/sigma', ExPwr.sigma);

Model.prior  = ExPwr;
Model.A = A;

[dA, A] = calcDeltaA (S, Model);

saveData (fd, '/test/ica/dA', dA);

epsilon = interpIter (1, iterPts, epsSteps);
eps = epsilon/max(abs(dA(:)));
dA = eps * dA;
Aref = A + dA;

saveData (fd, '/test/ica/eps', epsilon);
saveData (fd, '/test/ica/Aref', Aref);


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