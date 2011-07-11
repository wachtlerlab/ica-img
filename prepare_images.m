function [ loaded_images ] = prepare_images (DataPar)
%%
fprintf (['\n Loading images']);

Img.filename = '';
Img.refPath = '';
Img.imgPath = '';
Img.refpixelN = 0;
Img.refkoos = zeros (4, 1);
Img.imgData = NaN;
Img.SML = NaN;
Img.edgeN = 0;

nFiles = length (DataPar.fileList);
loaded_images = repmat (Img, nFiles, 1);

for idx=1:nFiles

    flnm = DataPar.fileList (idx,:);
    Img.filename = flnm;
    
    ref_path = strtrim ([DataPar.refDir, '/' , flnm]);
    Img.refPath = strjust ([ref_path , '.ref -ascii']);
    Img.imgPath = strtrim ([DataPar.dataDir, '/' flnm]);
    
    Img = load_image (Img, DataPar);
    loaded_images(idx) = Img;
    
end

fprintf ('\n Done loading images');
telapsed = toc;
fprintf ([' (', num2str(telapsed), ')\n']);

end


function [Img] = load_image (Img, DataPar)

flnm = Img.filename;
imgpath = Img.imgPath;
refpath = Img.refPath;

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

[n,T] = size (data);
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
Img.SML = datamx2;

if DataPar.doFilter
    Img = do_filter_image (Img, DataPar);
end


% how big is ref card
Img.refpixelN = (Img.refkoos(3)-Img.refkoos(1)+1)*(Img.refkoos(4)-Img.refkoos(2)+1);


end


function [Img] = do_filter_image (Img, DataPar)
%% Filter the data
flnm = Img.filename;
datamx2 = Img.imgData;

tic;
fprintf ([' applying filter to ' , flnm]);

ftData = feval (DataPar.filterFn, datamx2, DataPar);

telapsed = toc;
fprintf ([' (', num2str(telapsed), ')\n']);


%% Use the filtered data now
s = zeros(2, 3);
s(2,:) = size (ftData);
s(1,:) = size (datamx2);

sizeDiff = s(1,:) - s(2,:);

edgeN = s(2,2);

if edgeN ~= round (edgeN)
    warning ('Image after filtering not square!');
end

Img.edgeN = edgeN;

deltaM = sizeDiff(2);
deltaN = sizeDiff(3);

Img.refkoos(1) = Img.refkoos(1) - deltaM;
Img.refkoos(2) = Img.refkoos(2) - deltaN;

Img.imgData = ftData;

end
