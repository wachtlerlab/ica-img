function [ loaded_images ] = prepare_images (DataPar)
%%
fprintf (['\n Loading images']);

Img.filename = '';
Img.refkoos = zeros (4, 1);
Img.edgeN = 0;
Img.imgData = NaN;
Img.SML = NaN;
Img.refpixelN = 0;
Img.filtered = 0;

nFiles = length (DataPar.fileList);
loaded_images = repmat (Img, nFiles, 1);

for idx=1:nFiles

    flnm = DataPar.fileList (idx,:);
    Img = loadImage (flnm, DataPar.dataDir);
    
    if DataPar.doFilter
      Img = do_filter_image (Img, DataPar);
    end

    loaded_images(idx) = Img;
    
end

fprintf ('\n Done loading images');
telapsed = toc;
fprintf ([' (', num2str(telapsed), ')\n']);

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

% recalcuate how big the ref card is
Img.refpixelN = (Img.refkoos(3)-Img.refkoos(1)+1) * ...
  (Img.refkoos(4)-Img.refkoos(2)+1);

Img.filtered = 1;

end
