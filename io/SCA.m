classdef SCA < H5File
  %SCA Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function fh = SCA (filename)
      fh = fh@H5File([]);
      fh.fd = H5F.open (filename, ...
        'H5F_ACC_RDWR','H5P_DEFAULT');
    end
    
    function model = readModel (fh, modelid, cfgid)
      
      if nargin < 3
        g = fh.openGroup (['/ICA/']);
        l = g.listChildren();
        cfgid = l{1};
        g.close()
      end
      
      if nargin < 2
        g = fh.openGroup (['/ICA/' cfgid '/model']);
        l = g.listChildren();
        modelid = l{1};
        g.close()
      end
      
      path = ['/ICA/' cfgid '/model/' modelid];
      group = fh.openGroup (path);
      model.id = group.get ('id');
      model.cfg = group.get ('cfg');
      model.ds = group.get ('ds');
      model.creator = group.get ('creator');
      model.ctime = group.get ('ctime');
      model.fit_time = group.get ('fit_time');
      model.gpu = group.get('gpu');
      model.A = fh.read([path '/A']);
      model.beta = fh.read ([path '/beta']);
      group.close()
      
    end
    
    function cfg = readConfig (fh, cfgid)
      
      if nargin < 2
        g = fh.openGroup (['/ICA/']);
        l = g.listChildren();
        cfgid = l{1};
        g.close()
      end
      
      path = sprintf ('/ICA/%.7s', cfgid)
      group = fh.openGroup (path);
      text = fh.read ([path '/config']);
      cfg = loadjson(text');
      id = group.get ('id');
      cfg = setfield_idx (cfg, 'id', id, 1);
      group.close()
      
    end
    
    function [models] = listModel(fh, cfgid)
      
      if nargin < 2
        g = fh.openGroup (['/ICA/']);
        l = g.listChildren();
        cfgid = l{1};
        g.close()
      end
      
      g = fh.openGroup (['/ICA/' cfgid '/model/']);
      models = g.listChildren();
      g.close()
    end
    
  end
  
end

