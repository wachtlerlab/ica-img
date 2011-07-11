

dstr = datestr (clock (), 'yyyymmddHHMM');
filename = sprintf ('../results/%s-%s.mat', Model.name, dstr);
save (filename, 'Results', 'Model');
fprintf ('Saved results to %s\n', filename);
