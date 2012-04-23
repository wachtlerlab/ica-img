function [ output_args ] = saveConfig (cfg)
%SAVECONFIG Summary of this function goes here
%   Detailed explanation goes here

fd = H5F.create ('test_cfg.h5', 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');

cfg_id = cfg.id;

gcpl = H5P.create ('H5P_LINK_CREATE');
H5P.set_create_intermediate_group (gcpl, 1);
group = H5G.create (fd, ['/' cfg_id], gcpl,'H5P_DEFAULT', 'H5P_DEFAULT');

txt = savejson('', cfg);
cfg_text = regexprep(txt, '\t', '    ');
saveText (fd, [cfg_id '/config'], cfg_text);

H5G.close (group);
H5F.close (fd);

end

function [dsid] = saveText(fd, key, data)

if ischar(data)
  tmp = data;
  data = cell(1);
  data{1} = tmp;
end

dtype = H5T.copy ('H5T_C_S1');
H5T.set_size (dtype, 'H5T_VARIABLE');

% Create a dataspace for cellstr
H5S_UNLIMITED = H5ML.get_constant_value ('H5S_UNLIMITED');
dsp = H5S.create_simple (1, numel(data), H5S_UNLIMITED);

% Create a dataset plist for chunking
plist = H5P.create('H5P_DATASET_CREATE');
H5P.set_chunk(plist, 1); % 2 strings per chunk

% Create dataset
dset = H5D.create (fd, key, dtype, dsp, plist);

H5D.write (dset, dtype, 'H5S_ALL', 'H5S_ALL', 'H5P_DEFAULT', data);

if nargout < 1
  H5D.close(dset);
else
  dsid = dset;
end

H5S.close(dsp);
H5T.close(dtype);


end