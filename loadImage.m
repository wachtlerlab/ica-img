function [ Img ] = loadImage (filename, dataDir)
%loadImage Load an image (and it's ref card) from a file

if exist (dataDir, 'dir') ~= 7
  error ('dataDir non existent or not a directory');
end

flnm = strtrim (filename);
Img.filename = flnm;

imgpath = fullfile (dataDir, 'rad', flnm);
refpath = fullfile (dataDir, 'ref', [flnm '.ref']);


if exist ([imgpath '.mat'], 'file') == 0 || exist (refpath, 'file') == 0
  error ('Image data or refcard not found!\n\t[%s|%s]', imgpath, refpath);
end

fprintf ('\n Loading data for [%s] ', flnm);

%% load reference coords
fprintf (['\n\t [ref data] from ', refpath]);

% load coordinates of gray reference reflectance card
% from directory "ref/" next to data directory
eval (['load ', refpath, ' -ascii']);
Img.refkoos = eval (flnm);


%% Load image data into 'data' variable
fprintf (['\n\t [img data] from ', imgpath, '.mat']);
tic;
eval (['load ', imgpath]);
data = eval (flnm);
eval (['clear ', flnm]);

telapsed = toc;
fprintf (['\n\t Total time to load data: ', num2str(telapsed), '\n']);


%% Reshaping data
fprintf (' reshaping data ');
tic;

[~, T] = size (data);
edgeN = sqrt (T);

if edgeN ~= round (edgeN)
    error ('Image data to square!');
end

Img.edgeN = edgeN;

datamx = feval ('toSML', data);
% datamx2 = log(datamx2+0.01*max(datamx2(:)));
datamx2 = log (datamx); % offset not necessary - value 0 should not occur after SML trafo

% reshape to 256x256 pixels
datamxtmp = reshape (datamx2, 3, edgeN, edgeN);
datamx2 = datamxtmp;

% for k=1:31,
%  datamx(k,:,:)=datamx(k,:,:)-mean(datamx(k,:));
% end

telapsed = toc;
fprintf ([' (',num2str(telapsed),')\n']);

Img.imgData = datamx2;
Img.SML = permute (reshape (datamx, 3, edgeN, edgeN), [3 2 1]);

% how big is ref card
Img.refpixelN = (Img.refkoos(3)-Img.refkoos(1)+1)*(Img.refkoos(4)-Img.refkoos(2)+1);

Img.filtered = 0;

end

