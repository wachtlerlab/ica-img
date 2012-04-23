function [ cfg ] = loadConfig (configId)

configFile = fullfile ('config', [configId '.json']);

if exist (configFile, 'file') == 0
  error ('Cannot find config');
end

cfg = loadjson(configFile);

git_rev = getCurRev();
mat_ver = regexprep(version, ' \(.+\)', '');

creator = ['MATLAB ' mat_ver ' [' git_rev(1:7) ']'];
cfg = setfield_idx (cfg, 'creator', creator, 'version');


id = createCfgId (cfg);
cfg = setfield_idx (cfg, 'id', id, 1);

id_check =  createCfgId (cfg);

if ~strcmpi (id, id_check)
  warning ('ica:config_id', 'config id (SHA-1) is not stable');
end

end

function [id] = createCfgId (cfg)

if isfield (cfg, 'id')
  cfg = rmfield (cfg, 'id');
end

%canonicalize the (textual) config
txt = savejson('', cfg);
cfg_text = regexprep(txt, '\t', '    ');

id = DataHash (cfg_text, struct ('Method', 'SHA-1'));

end

function [S] = setfield_idx (S, key, value, idx)

S.(key) = value;

if ischar(idx)
  idx = findfield(S, idx) + 1;
end

cols =1:length(fieldnames (S)); 
S = orderfields (S, [cols(1:idx-1), cols(end), cols(idx:end-1)]);

end

function [idx] = findfield(S, name)

idx = 0;
fields = fieldnames (S);

for n = 1:length(fields)
  if strcmpi (fields(n), name)
    idx = n;
    break;
  end
end

end