function [ value ] = getDefaults (key)

defaultsFile = '~/.matlab.defaults';

persistent defaults;
persistent mtime;

lstat = dir(defaultsFile);

if isempty(defaults) || lstat.datenum > mtime
    defaults = loadjson(defaultsFile);
    mtime = lstat.datenum;
end

if isempty(key)
    return;
end

treepath = regexp(key,'\.','split');
value = getfield(defaults, treepath{:});

end

