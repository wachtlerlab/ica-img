function [ Img ] = loadImage (filename, dataDir, imageType)

if strncmp(imageType, 'noref', 5)
    imageType = imageType(7:end);
    Img = loadImgNoRef(filename, dataDir, imageType);
else
    Img = loadImgRef(filename, dataDir, imageType);
end

end

function [ Img ] = loadImgRef(filename, dataDir, imageType)
%loadImage Load an image (and it's ref card) from a file

if nargin < 3
    imageType = 'rad';
end

if exist (dataDir, 'dir') ~= 7
  error ('dataDir non existent or not a directory');
end

flnm = strtrim (filename);
Img.filename = flnm;

imgpath = fullfile (dataDir, imageType, flnm);
refpath = fullfile (dataDir, 'ref', [flnm '.ref']);


if exist ([imgpath '.mat'], 'file') == 0 || exist (refpath, 'file') == 0
  error ('Image data or refcard not found!\n\t[%s|%s]', imgpath, refpath);
end

fprintf ('\n Loading data for [%s] ', flnm);

%% load reference coords
fprintf (['\n\t [ref data] from ', refpath]);

% load coordinates of gray reference reflectance card
% from directory "ref/" next to data directory

% the format of the refcard is 
% x_start y_start x_end y_end

eval (['load ', refpath, ' -ascii']);
Img.refkoos = eval (flnm);


%% Load image data into 'data' variable
fprintf (['\n\t [img data] from ', imgpath, '.mat']);
tic;
eval (['load ', imgpath]);
data = eval (flnm);
eval (['clear ', flnm]);

Img.hs_data = data;

telapsed = toc;
fprintf (['\n\t Total time to load data: ', num2str(telapsed), '\n']);

Img.filtered = 0;
end

function [ Img ] = loadImgNoRef(filename, dataDir, imageType)
%loadImage Load an image (and it's ref card) from a file

if nargin < 3
    imageType = 'rad';
end

if exist (dataDir, 'dir') ~= 7
  error ('dataDir non existent or not a directory');
end

flnm = strtrim (filename);
Img.filename = flnm;

imgpath = fullfile (dataDir, imageType, flnm);

if exist ([imgpath '.mat'], 'file') == 0
  error ('Image data not found!\n\t[%s]', imgpath);
end

fprintf ('\n Loading data for [%s] ', flnm);

%% Load image data into 'data' variable
fprintf (['\n\t [img data] from ', imgpath, '.mat']);

filevars = whos ('-file', imgpath);

tic;
eval (['load ', imgpath]);
data = eval (filevars(1).name);
eval (['clear ', flnm]);

Img.hs_data = data;

telapsed = toc;
fprintf (['\n\t Total time to load data: ', num2str(telapsed), '\n']);

Img.filtered = 0;
end
