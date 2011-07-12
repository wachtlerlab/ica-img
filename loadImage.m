function [ Img ] = loadImage (filename, dataDir)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

flnm = strtrim (filename);
Img.filename = flnm;
imgpath = fullfile (dataDir, 'rad', flnm);
refpath = fullfile (dataDir, 'ref', [flnm '.ref -ascii']);

%% load reference coords
fprintf (['\n Loading ', refpath, ' refcard data']);

% load coordinates of gray reference reflectance card
% from directory "ref/" next to data directory
eval (['load ', refpath]);
Img.refkoos = eval (flnm);


%% Load image data into 'data' variable
fprintf (['\n Loading ', flnm, ' image data']);
tic;
eval (['load ', imgpath]);
data = eval (flnm);
eval (['clear ', flnm]);

telapsed = toc;
fprintf ([' (', num2str(telapsed), ')\n']);


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
Img.SML = permute (datamx2, [3 2 1]);

% how big is ref card
Img.refpixelN = (Img.refkoos(3)-Img.refkoos(1)+1)*(Img.refkoos(4)-Img.refkoos(2)+1);

Img.filtered = 0;

end

