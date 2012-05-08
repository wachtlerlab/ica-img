classdef H5DataSet < H5Loc
  
  methods
    function ds = H5DataSet(handle)
      ds.handle = handle;
    end
    
    function close(ds)
      H5D.close(ds.handle);
      ds.handle = NaN;
    end
    
    function [value] = read(ds)
      value = H5D.read (ds.handle);
    end
    
  end
  
end
