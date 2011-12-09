classdef CUDA < handle
  
  properties
    h
  end
  
  methods
    
    function setup(cuda)
      %
      if ~libisloaded('')
        [notfound, warnings] = loadlibrary('icudamat',...
          'icudamat.h');
        
        cuda.h = calllib('icudamat', 'icudamat_create')
        
      end
    end
    
    function shutdown(cuda)
      unloadlibrary('icudamat');
    end
    
    function idx = idamax(cuda, A)
      idx = calllib('icudamat', 'icudamat_idamax', cuda.h, A)
    end
    
    function C = gemm(cuda, C, A, B, alpha, beta)
      ok = calllib('icudamat', 'icudamat_dgemm', cuda.h, A, 0, B, 0, C, alpha, beta);
    end
    
  end
  
end

