function [id] = createCfgId (cfg)

if isfield (cfg, 'id')
  cfg = rmfield (cfg, 'id');
end

if isfield (cfg, 'name')
  cfg = rmfield (cfg, 'name');
end

if isfield (cfg, 'creator')
  cfg = rmfield (cfg, 'creator');
end

if isfield (cfg, 'ctime')
  cfg = rmfield (cfg, 'ctime');
end


%canonicalize the (textual) config
txt = savejson('', cfg);
cfg_text = regexprep(txt, '\t', '    ');

id = DataHash (cfg_text, struct ('Method', 'SHA-1'));

end