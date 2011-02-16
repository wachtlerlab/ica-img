function [D] = mkpreprocpatch_plain(ddir,flnm,npats);
% mkcolordata -- 
%   Usage
%     [x] = mkpreprocpatch_plain(ddir,flnm,T)
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

fprintf(['\n Loading ', flnm, ' refcard data']);
% load coordinates of gray reference reflectance card
% from directory "ref/" next to data directory
eval(['load ', strjust([ddir, '/../ref/', flnm]), '.ref -ascii']);
refkoos=eval([flnm]);
% how big is ref card
refpixelN=(refkoos(3)-refkoos(1)+1)*(refkoos(4)-refkoos(2)+1);


fprintf(['\n Loading ', flnm, ' image data']);
tic;
eval(['load ', ddir, '/', flnm]);

telapsed=toc;
fprintf([' (',num2str(telapsed),')\n randomizing indexes']);
tic;

[n, T] = size(eval([flnm]));

patchsize=7;   % size of image patch
patchradius=(patchsize-1)/2; % patchsize should be odd number
patchpixelN=patchsize^2;
spectbandN=3;  % how many wavebands
spectavgN=1;   % how many wavelengths to average for each spectband
spectoffset=1; % how many wavelengths to skip at short wavelength end
ergvecdim=patchpixelN*spectbandN;

pixelN=256^2;         % # pixel in image
innerN=(256-(patchsize-1))^2; % # of pixel not at edge of image
inneridx=zeros(innerN,2);
excl=[];
cnt=1;
for i=1:(256-patchsize),
  for j=1:(256-patchsize),
    if ((i>refkoos(1)-patchsize)&(i<refkoos(3))&(j>refkoos(2)-patchsize)&(j<refkoos(4))),
     excl=[excl [i;j]];
    else
     inneridx(cnt,1) = i; 
     inneridx(cnt,2) = j; 
     cnt=cnt+1;
    end
  end
end
cnt=cnt-1; % set cnt to max index value
eval(['save excluded_',flnm,' excl']); 
inneridx=inneridx(1:cnt,:);
inneridxidx=randperm(cnt);
% patchidx=zeros(innerN,2);
% for i=1:innerN,
%  patchidx(i,1) = inneridx(inneridxidx(i),1);
%  patchidx(i,2) = inneridx(inneridxidx(i),2); 
% end
patchidx=inneridx(inneridxidx,:);

telapsed=toc;
fprintf([' (',num2str(telapsed),')\n reshaping data ']);
tic;

% originally: datamx=reshape(eval([flnm]),31,256,256), but see new toSML
datamx=reshape(eval([flnm]),31,256*256);

datamx2=feval('toSML',datamx);
% datamx2=log(datamx2+0.01*max(datamx2(:)));
datamx2=log(datamx2); % offset not necessary - value 0 should not occur after SML trafo

% reshape to 256x256 pixels
datamxtmp=reshape(datamx2,3,256,256);
datamx2=datamxtmp;


% for k=1:31,
%  datamx(k,:,:)=datamx(k,:,:)-mean(datamx(k,:));
% end

telapsed=toc;
fprintf([' (',num2str(telapsed),')\n extracting patches']);
tic;

uh = T - 1;
iidx = randperm(npats);
xidx = 1 + ceil(uh*rand(npats,1));

D = zeros(ergvecdim,npats);
patch = zeros(n,1);

datapatch=zeros(patchpixelN,1);
for j=1:npats,
  ix = patchidx(j,1); 
  iy = patchidx(j,2); 
  datapatch=datamx2(:,ix:ix+patchsize-1,iy:iy+patchsize-1);
  D(:,j) = reshape(datapatch,ergvecdim,1);
end

telapsed=toc;
fprintf([' (',num2str(telapsed),')']);


