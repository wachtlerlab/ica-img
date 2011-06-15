function [D] = generate_patches (Img, npats, DataPar)

edgeN = Img.edgeN;
refkoos = Img.refkoos;
flnm = Img.filename;

tic;
fprintf (' randomizing indexes ');

patchsize   = DataPar.patchSize; % size of image patch
patchradius = (patchsize-1)/2;   % patchsize should be odd number
patchpixelN = patchsize^2;
spectbandN  = 3;                 % how many wavebands
spectavgN   = 1;                 % how many wavelengths to average for each spectband
spectoffset = 1;                 % how many wavelengths to skip at short wavelength end
ergvecdim   = patchpixelN * DataPar.dataDim;

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
fprintf(' extracting patches');
tic;


%T = Img.edgeN^2;
%uh = T - 1;
%iidx = randperm (npats);
%xidx = 1 + ceil (uh*rand (npats, 1));

D = zeros (ergvecdim, npats);

datapatch=zeros (patchpixelN, 1);

for j = 1:npats,
  ix = patchidx (j,1); 
  iy = patchidx (j,2); 
  datapatch = Img.imgData (:, ix:ix+patchsize-1, iy:iy+patchsize-1);
  D(:,j) = reshape (datapatch, ergvecdim, 1);
end

telapsed = toc;
fprintf ([' (', num2str(telapsed), ') \n']);

end

