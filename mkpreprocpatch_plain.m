function [D] = mkpreprocpatch_plain (ddir, flnm, npats)
% mkcolordata -- 
%   Usage
%     [x] = mkpreprocpatch_plain (ddir, flnm, T)
%   Inputs
%          ddir : path to image files
%          flnm : name of file
%         npats : number of samples
%   Outputs
%             x : randomly sampled data
%
%
% improved log 
%

% for debugging purposes only
figure('Name', flnm);
%set (0, 'CurrentFigure', 1);

%% load reference coords
fprintf (['\n Loading ', flnm, ' refcard data']);

% load coordinates of gray reference reflectance card
% from directory "ref/" next to data directory
eval (['load ', strjust([ddir, '/../ref/', flnm]), '.ref -ascii']);
refkoos = eval(flnm);

% how big is ref card
refpixelN = (refkoos(3)-refkoos(1)+1)*(refkoos(4)-refkoos(2)+1);


%% Load image data into 'data' variable
fprintf (['\n Loading ', flnm, ' image data']);
tic;
eval (['load ', ddir, '/', flnm]);
data = eval (flnm);
eval (['clear ', flnm]);

telapsed = toc;
fprintf ([' (', num2str(telapsed), ')\n']);


%% Reshaping data
fprintf ('reshaping data ');
tic;

[n,T] = size (data);
edgeN = sqrt (T);

if edgeN ~= round (edgeN)
    error ('Image data to square!');
end

% originally: datamx=reshape(eval([flnm]),31,256,256), but see new toSML
%datamx = reshape (data, 31, edgeN * edgeN);

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

%% Applying Filter
tic;
fprintf (['Applying filter to ' , flnm]);

mhat = mexican_hat (3, 20, 1, 6);
ftData(1,:,:) = conv2 (squeeze(datamx2(1,:,:)), mhat, 'valid');
ftData(2,:,:) = conv2 (squeeze(datamx2(2,:,:)), mhat, 'valid');
ftData(3,:,:) = conv2 (squeeze(datamx2(3,:,:)), mhat, 'valid');

telapsed = toc;
fprintf ([' (', num2str(telapsed), ')\n']);

%% debug info
refxs = [1, 3, 3, 1, 1];
refys = [2, 2, 4, 4, 2];

fprintf ([' (',num2str(telapsed),')\n']);

for idximg = 1:3
   subplot (2, 3, idximg);
   slice = squeeze (datamx2 (idximg, :, :));
   imagesc (slice');
   hold on;
   colormap ('gray');
   axis image
   axis off
   plot (refkoos(refxs), refkoos(refys), 'r-');
end

drawnow;

%% Use the filtered data now
s = zeros(2, 3);
s(2,:) = size (ftData);
s(1,:) = size (datamx2);

sizeDiff = s(1,:) - s(2,:);

edgeN = s(2,2);

if edgeN ~= round (edgeN)
    warning ('Image after filtering not square!');
end

deltaM = sizeDiff(2);
deltaN = sizeDiff(3);

refkoos(1) = refkoos(1) - deltaM;
refkoos(2) = refkoos(2) - deltaN;

datamx2 = ftData;

%% debugging

for idximg = 1:3
   subplot (2, 3, idximg + 3);
   slice = squeeze (datamx2 (idximg, :, :));
   imagesc (slice');
   hold on;
   colormap ('gray');
   axis image
   axis off
   plot (refkoos(refxs), refkoos(refys), 'r-');
end

drawnow;


%% Create and randomize indexes
tic;
fprintf ('randomizing indexes ');

patchsize   = 7;               % size of image patch
patchradius = (patchsize-1)/2; % patchsize should be odd number
patchpixelN = patchsize^2;
spectbandN  = 3;               % how many wavebands
spectavgN   = 1;               % how many wavelengths to average for each spectband
spectoffset = 1;               % how many wavelengths to skip at short wavelength end
ergvecdim   = patchpixelN*spectbandN;

pixelN   = edgeN * edgeN;                       % # pixel in image
innerN   = (edgeN - (patchsize-1))^2; % # of pixel not at edge of image
inneridx = zeros (innerN, 2);
excl = [];
cnt = 1;

for i=1:(edgeN-patchsize),
  for j=1:(edgeN-patchsize),
    if ((i>refkoos(1)-patchsize)&(i<refkoos(3))&(j>refkoos(2)-patchsize)&(j<refkoos(4))),
     excl=[excl [i;j]];
    else
     inneridx(cnt,1) = i; 
     inneridx(cnt,2) = j; 
     cnt = cnt+1;
    end
  end
end

eval (['save excluded_',flnm,' excl']);

cnt = cnt - 1;                   % set cnt to max index value 
inneridx = inneridx (1:cnt, :);
inneridxidx = randperm (cnt);

% patchidx=zeros(innerN,2);
% for i=1:innerN,
%  patchidx(i,1) = inneridx(inneridxidx(i),1);
%  patchidx(i,2) = inneridx(inneridxidx(i),2); 
% end

patchidx = inneridx (inneridxidx, :);

telapsed = toc;
fprintf ([' (',num2str(telapsed),')\n']);

%% Extracting patches
fprintf(['extracting patches']);
tic;

uh = T - 1;
iidx = randperm (npats);
xidx = 1 + ceil (uh*rand (npats, 1));

D = zeros (ergvecdim, npats);
patch = zeros (n,1);

datapatch=zeros (patchpixelN,1);
for j = 1:npats,
  ix = patchidx (j,1); 
  iy = patchidx (j,2); 
  datapatch=datamx2 (:, ix:ix+patchsize-1, iy:iy+patchsize-1);
  D(:,j) = reshape (datapatch, ergvecdim, 1);
end

telapsed = toc;
fprintf ([' (', num2str(telapsed), ') \n']);


