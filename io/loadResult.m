function [ Model] = loadResult(id, mid)

cfg = [];

path = fullfile('~','Coding','ICA','data');
  
if (nargin < 1)

  files = dir(path);

  dates = arrayfun (@listfilter, files);
  [~,idx] = sort(dates, 'descend');
  
  reply = input_choose_file(files(idx(1:5)));
  
  filename = files(idx(reply)).name;
  
else
  
  pattern = fullfile(path, ['*' id '*']);
  files = dir (pattern);
  idx = 1;
  
  if isempty (files)
    fprintf ('No such result found!');
    Model = [];
    return
  elseif length(files) > 1
    fprintf ('Input not unique');
    idx = input_choose_file (files);
  end
  
  filename = files(idx).name;
  
end

filepath = fullfile (path, filename);
  
sca = SCA(filepath);
if nargin > 1
  Model = sca.readModel (mid);
else
  Model = sca.readModel ();
end

cfg = sca.readConfig ();
ds = sca.readDataset(Model.cfg, Model.ds);

Model.cfg = cfg;
Model.ds = sca.readDataset(cfg.id, ds.id);

sca.close()

end

function [dn] = listfilter(p)

if p.isdir
  dn = 0;
else
  dn = p.datenum;
end

end

function [res] = hassuffix(str, suffix)
res = strcmpi (str(end-(length(suffix)-1):end), suffix);
end


function [reply] = input_choose_file (files)

for n = 1:5
  fprintf ('%d: %s\n', n, files(n).name)
end

reply = input('Which result to load? [1]: ', 's');
if isempty(reply)
  reply = 1;
else
  reply = str2num(reply);
end

end