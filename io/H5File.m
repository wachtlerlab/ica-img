classdef H5File < handle
  %H5FILE HDF5 I/O class
  %   
  
  properties(Access = protected)
    fd = NaN;
  end
  
  methods
   
    function fh = H5File(filename)
      if length (filename) > 0
        fh.fd = H5F.create (filename, ...
          'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');
      end
      
    end
    
    function close(fh)
      H5F.close (fh.fd);
    end
    
    function [group] = createGroup(fh, location)
      gcpl = H5P.create ('H5P_LINK_CREATE');
      H5P.set_create_intermediate_group (gcpl, 1);
      gh = H5G.create (fh.fd, location, gcpl,'H5P_DEFAULT', 'H5P_DEFAULT');
      group = H5Group(gh);
    end
    
    function [group] = openGroup(fh, location)
      gh = H5G.open (fh.fd, location);
      group = H5Group(gh);
    end
    
    function [dset] = openDataSet(fh, location)
      dsid = H5D.open(fh.fd , name);
      dset = H5DataSet (dsid);
    end
    
    function [dsid] = write(fh, loc, data)
      
      if ischar(data)
        writer = @fh.writeText;
      elseif isnumeric(data)
        writer = @fh.writeNumeric;
      else
        error ('Unsupported data type')
      end
      
      if nargout < 1
        writer(loc, data);
      else
        dsid = writer(loc, data);
      end
      
    end
    
    function [dsid] = writeText(fh, loc, data)
      
      dtype = H5T.copy ('H5T_C_S1');
      H5T.set_size (dtype, numel(data));
      
      dsp = H5S.create_simple (1, 1, []);
      
      % Create a dataset plist for chunking
      plist = H5P.create('H5P_DATASET_CREATE');
      H5P.set_chunk(plist, 1);
      
      % Create dataset
      dset = H5D.create (fh.fd, loc, dtype, dsp, plist);
      
      H5D.write (dset, 'H5ML_DEFAULT', 'H5S_ALL', 'H5S_ALL', 'H5P_DEFAULT', data);
      
      if nargout < 1
        H5D.close(dset);
      else
        dsid = H5DataSet (dset);
      end
      
      H5P.close(plist);
      H5S.close(dsp);
      H5T.close(dtype);
    end
   
    function [dsid] = writeNumeric (fh, name, data)
      
      dims = fliplr (size (data));
      dsp = H5S.create_simple (length(dims), dims, []);
      h5klass = class2h5(data);
      dtype = H5T.copy(h5klass);
      
      dset = H5D.create(fh.fd, name, dtype, dsp, 'H5P_DEFAULT');
      H5D.write (dset, 'H5ML_DEFAULT','H5S_ALL','H5S_ALL', ...
        'H5P_DEFAULT', data);
      
      if nargout < 1
        H5D.close(dset);
      else
        dsid = H5DataSet (dset);
      end
      
      H5S.close(dsp);
      H5T.close(dtype);
    end
    
    function [value] = read(fh, name)
      dset = H5D.open(fh.fd , name);
      value = H5D.read(dset);
      H5D.close (dset);
    end  
      
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
