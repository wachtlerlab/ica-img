function Result = samplePats_plain(Result,fitPar,DataPar)
% return a unique sample of patterns

% Written by Mike Lewicki 4/99
%
% Copyright (c) 1999 Michael S. Lewicki and CMU
%           (c) 2011 Christian Kellner
%
% Permission to use, copy, modify, and distribute this software and its
% documentation for any purpose and without fee is hereby granted,
% provided that the above copyright notice and this paragraph appear in
% all copies.  Copyright holder(s) make no representation about the
% suitability of this software for any purpose. It is provided "as is"
% without express or implied warranty.


Nimgs=length (Result.images);      % # images in list
Firstimg=1;    % first and last images to use out of list above
Lastimg=Nimgs; %

Tperimg=5000;
T = Nimgs*Tperimg;


if (isempty(Result.X) || Result.dataIdx > fitPar.npats)
  % generate a new dataset

  fprintf('%5d: Generating new images patches\n',Result.iter);
    
  % load new data
    xx=[];
    for idx=Firstimg:Lastimg
        img = Result.images(idx);
        
        xtmp = generate_patches (img, Tperimg, DataPar);
     
        % xtmp = xtmp - (ones(Tperimg,1)*mean(xtmp'))';
        xtmp = xtmp - mean(xtmp(:));
        xtmp=xtmp/sqrt(var(xtmp(:)));
        xx=[ xx xtmp ];
    end
    
    permidxlst=randperm(T);
    x=xx(:,permidxlst);
    x = x - (ones(T,1)*mean(x'))';
    x=x/sqrt(var(x(:)));
 
  % Result.X = feval(fitPar.dataFn, fitPar.dataFnArgs);
  Result.X = x;
  
  Result.dataIdx = 1;
end


if Result.dataIdx + fitPar.blocksize -1 >= fitPar.npats
  r = ceil(fitPar.npats*rand(fitPar.blocksize,1));
  Result.D = Result.X(:,r);
else
  xs = Result.dataIdx;
  xe = xs + fitPar.blocksize - 1; 
  Result.D = Result.X(:,xs:xe);
end

Result.dataIdx = Result.dataIdx + fitPar.blocksize;
