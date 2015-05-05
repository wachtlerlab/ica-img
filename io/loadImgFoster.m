function [ Img ] = loadImgFoster(filepath, dataDir, source)

filepath = strtrim (filepath);

idx = strfind(filepath, '@');
flnm = filepath(1:idx-1);
illu = filepath(idx+1:end);

imgpath = fullfile (dataDir, 'foster', '2002', flnm);

fprintf ('\n Loading data for [%s] ', flnm);

%% Load image data into 'data' variable
fprintf (['\n\t [img data] from ', imgpath, '.mat']);

filevars = whos ('-file', imgpath);

tic;
eval (['load ', imgpath]);
imgdata = eval (filevars(1).name);
eval (['clear ', flnm]);


%% Load the illuminant
ilupath = fullfile (dataDir, 'foster', '2002', ['illum_' illu]);
filevars = whos ('-file', ilupath);

eval (['load ', ilupath]);
iludata = eval (filevars(1).name);
eval (['clear ', flnm]);

[m, n, c] = size(imgdata);
rf = reshape (imgdata, m*n,c);
data = rf'.*repmat(iludata(2:32), 1, m*n);

imgdata = reshape(data, 31, m, n);
imgdata = permute(imgdata, [1 3 2]);
imgdata = flipdim(imgdata, 2);

data = reshape(imgdata, 31, m*n);

Img.hs_data = data;
Img.filename = flnm;

end

