function [cfg] = loadConfig(base, varargin)

if isempty(base)
  cfg = struct();
elseif isstruct(base)
  cfg = base;
elseif ischar(base)
  cfg = loadjson (['config/' base '.json']);
end

cfg = processImports(cfg);

args = size (varargin);
for kv = 1:2:args(2)
  arg = varargin{kv};
  val = varargin{kv + 1};
  
  if ischar (val)
    %assume it is a file to load
    override = loadjson(val);
  elseif isstruct (val)
    override = val;
  end
  
  loc = regexp (arg, '\.', 'split');
  target = getfield (cfg, loc{:});
  
  if isstruct(target)
    override = mergeStructs(target, override);
  end
  
  cfg = setfield(cfg, loc{:}, override);
end

git_rev = getCurRev();
mat_ver = regexprep(version, ' \(.+\)', '');


id = createCfgId (cfg);
cfg = setfield_idx (cfg, 'id', id, 1);

id_check =  createCfgId (cfg);

creator = ['MATLAB ' mat_ver ' [' git_rev(1:7) ']'];
cfg = setfield_idx (cfg, 'creator', creator, 'version');
cfg = setfield_idx (cfg, 'ctime', gen_ctime(), 'creator');

if ~strcmpi (id, id_check)
  warning ('ica:config_id', 'config id (SHA-1) is not stable');
end

end

function [S] = processImports (S)

%process nested structs first
names = fieldnames (S);
for n = 1:length(names)
  key = char (names(n));
  
  if isstruct (S.(key))
    S.(key) = processImports (S.(key));
  end
  
end

if isfield (S, 'x_0x23_import')
  import = S.x_0x23_import;
  idx = findfield(S, 'x_0x23_import');
  S = rmfield (S, 'x_0x23_import');
  merge = loadjson(['config/' import '.json']);
  N = length(fieldnames(S));
  M = length(fieldnames(merge));
  S = mergeStructs (S, merge);
  
  cols = 1:length(fieldnames (S)); 
  seq = [cols(1:idx-1), cols(N+1:(N+M)), cols(idx:N)];
  S = orderfields (S, seq);
  
  S = processImports (S);  % we might have imported a new import
end

end

function [id] = createCfgId (cfg)

if isfield (cfg, 'id')
  cfg = rmfield (cfg, 'id');
end

if isfield (cfg, 'name')
  cfg = rmfield (cfg, 'name');
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