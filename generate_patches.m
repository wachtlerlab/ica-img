function [D] = generate_patches (Img, npats, DataPar)

patchsize   = DataPar.patchSize; 
patchpixelN = patchsize^2;
ergvecdim   = patchpixelN * DataPar.dataDim;

fprintf ('\t Image: %s\n', Img.filename);
fprintf ('\t\t randomizing indexes\t');

tic;
cnt = size (Img.refBase, 1);

inneridxidx = randperm (cnt);
patchidx = Img.refBase (inneridxidx, :);

telapsed = toc;
fprintf ([' (',num2str(telapsed),')\n']);

%% Extracting patches
fprintf('\t\t extracting patches\t');
tic;

D = zeros (ergvecdim, npats);

datapatch = zeros (patchpixelN, 1);

for j = 1:npats,
  ix = patchidx (j,1); 
  iy = patchidx (j,2); 
  datapatch = Img.imgData (:, ix:ix+patchsize-1, iy:iy+patchsize-1);
  D(:,j) = reshape (datapatch, ergvecdim, 1);
end

telapsed = toc;
fprintf ([' (', num2str(telapsed), ') \n']);

end

