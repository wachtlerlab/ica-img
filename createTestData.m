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
H5F.close (fd);

end

function [dsid] = saveData (fd, name, data)

dims = fliplr (size (data));
dsp = H5S.create_simple (length(dims), dims, dims);
dtype = H5T.copy('H5T_NATIVE_DOUBLE');

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