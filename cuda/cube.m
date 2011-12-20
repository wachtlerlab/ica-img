classdef cube < handle
  
  properties
    h
  end
  
  methods
    
    function setup(cube)
      %
      if ~libisloaded('')
        [notfound, warnings] = loadlibrary('libcube_m',...
          'cube_matlab.h');
        
        [notfound, warnings] = loadlibrary('libcube',...
          'cube.h');
        
        cube.h = calllib('libcube', 'cube_context_new')
        
      end
    end
    
    function nA = ica_update_A(cube, A, S, mu, beta, sigma, epsilon)
      nA = calllib('libcube_m', 'cube_matlab_ica_update_A', cube.h, A, S, mu, beta, sigma, epsilon);
    end
    
  end
  
end

