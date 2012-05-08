classdef H5Group < H5Loc
  
  methods
    function group = H5Group(handle)
      group.handle = handle;
    end
    
    function close(group)
      H5G.close(group.handle);
      group.handle = NaN;
    end
    
  end
  
end