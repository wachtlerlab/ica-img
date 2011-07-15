function [filename] = saveResult (Model)

dstr = datestr (clock (), 'yyyymmddHHMM');
filename = sprintf ('../results/%s-%s.mat', Model.name, dstr);
save (filename, 'Model');

end