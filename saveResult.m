function [filename] = saveResult (Model)

if isfield (Model, 'id')
  nick = Model.id(1:7);
else
  nick = datestr (clock (), 'yyyymmddHHMM');
end

filename = sprintf ('../results/%s-%s.mat', Model.name, nick);
save (filename, 'Model');

end