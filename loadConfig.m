function [ Model, FitParam, DisplayParam, DataParam ] = loadConfig (configId)
%loadConfig reads in config from simulation config file

configFile = sprintf('config/%s.m', configId);

fd = fopen(configFile);
data = textscan (fd,'%s','Delimiter','\n');
data = data{1};
fclose (fd);

codeData = sprintf('%s\n', data{:});
eval (codeData);

if isempty (Model) || ...
   isempty (FitParam) || ...
   isempty (DisplayParam) || ...
   isempty (DataParam)
  error ('Config file corrupt');
end

hashOpts.Method = 'SHA-1';
hashOpts.isFile = 'true';

Model.cfgId = DataHash (configFile, hashOpts);
Model.name = configId;


end

