classdef H5Loc
  
  properties (Access = protected)
    handle = NaN;
  end
  
  methods
    function set(loc, name, value)
      if isnumeric(value)
        writer = @loc.setNumeric;
      elseif ischar(value)
        writer = @loc.setText;
      else
        error ('Unsupported type')
      end
      
      writer(name, value);
      
    end
    
    function setNumeric(loc, name, value)
      
      space = H5S.create('H5S_SCALAR');
      dtype = class2h5(value);
      attr = H5A.create (loc.handle, name, dtype, space, 'H5P_DEFAULT');
      H5A.write (attr, 'H5ML_DEFAULT', value(1));
      H5A.close (attr);
      H5S.close (space);
      
    end
    
    function setText(loc, name, value)
      space = H5S.create('H5S_SCALAR');
      dtype = H5T.copy (class2h5(value));
      H5T.set_size (dtype, numel(value));
      attr = H5A.create (loc.handle, name, dtype, space, 'H5P_DEFAULT');
      H5A.write (attr, 'H5ML_DEFAULT', value);
      H5A.close (attr);
      H5S.close (space);
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
    
  case 'char'
    dtype = 'H5T_C_S1'; %FIXME: FORTRAN?
    
  otherwise
    error ('Implement me!');
end

end