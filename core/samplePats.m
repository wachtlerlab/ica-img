function [ Result ] = samplePats(Result, dataset)

blocksize = dataset.blocksize;
npats = dataset.npats;

iter = Result.iter;
blocks = npats / blocksize;                % blocks in cluster
block = mod (iter - 1 , blocks) + 1;       % current block in cluster
cluster = ((iter - block) / blocks) + 1;   % current cluster
dataIdx = ((block - 1) * blocksize) + 1;   % current data in the cluster

if (block == 1)
  tstart = tic;
  fprintf('%5d: Extracting new images patches\n', Result.iter);
  Result.X = extractPatches(dataset, cluster);
  fprintf('%5s  done in %f \n', ' ', toc(tstart));
end

if dataIdx + blocksize - 1 > npats
  r = ceil(npats*rand(blocksize,1));
  Result.D = Result.X(:,r);
  fprintf ('WARNING: running out of patches! (%d)\n', dataIdx);
else
  xs = dataIdx;
  xe = xs + blocksize - 1; 
  Result.D = Result.X(:,xs:xe);
end

end

