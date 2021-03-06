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
      
      path = ['/ICA/' cfgid '/model/' modelid '/']
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
        g = fh.openGroup ('/ICA/');
        l = g.listChildren();
        cfgid = l{1};
        g.close()
      end
      
      path = sprintf ('/ICA/%.7s', cfgid);
      group = fh.openGroup (path);
      text = fh.read ([path '/config']);
      cfg = loadjson(text');
      id = group.get ('id');
      cfg = setfield_idx (cfg, 'id', id, 1);
      group.close()
      
    end
    
    function ds = readDataset (fh, cfgid, dsid)
        
      path = sprintf ('/ICA/%.7s/dataset/%.7s', cfgid, dsid);
      group = fh.openGroup (path);
      
      ds.id = group.get ('id');
      ds.cfg = group.get ('cfg');
      ds.creator = group.get ('creator');
      ds.channels = group.get('channels');
      ds.blocksize = group.get('blocksize');
      ds.dim = group.get('dim');
      ds.npats = group.get('npats');
      ds.patchsize = group.get('patchsize');
      ds.nclusters = group.get('nclusters');
      ds.rng.seed = group.get('rng.seed');
      ds.rng.state = group.get('rng.state');
      ds.rng.type = group.get('rng.type');
      
      ds.indicies = fh.read([path '/indicies']);
      ds.patsperm = fh.read([path '/patsperm']);
      ds.imgdata = fh.read([path '/imgdata']);
      ds.Aguess = fh.read([path '/A_guess']);
      
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
    
    
    function saveDataSet(fd, dataset)
                 
      ds_id = dataset.id;
      loc = sprintf ('/ICA/%.7s/dataset/%.7s', dataset.cfg, ds_id);
      
      group = fd.createGroup(loc);
      group.set ('id', ds_id);
      group.set ('cfg', dataset.cfg);
      group.set ('creator', dataset.creator);
      group.set ('dim', int32(dataset.dim));
      group.set ('patchsize', int32(dataset.patchsize));
      group.set ('channels', dataset.channels);
      group.set ('npats', int32(dataset.npats));
      group.set ('blocksize', int32(dataset.blocksize));
      group.set ('nclusters', int32(dataset.nclusters));
      group.set ('maxiter', int32(dataset.maxiter));
      group.set ('rng', dataset.rng);
      
      group.close();
      
      fd.write ([loc '/indicies'], dataset.indicies);
      fd.write ([loc '/patsperm'], dataset.patsperm);
      fd.write ([loc '/imgdata'], dataset.imgdata);
      fd.write ([loc '/A_guess'], dataset.Aguess);
      
    end
    
    
  end
  
end

