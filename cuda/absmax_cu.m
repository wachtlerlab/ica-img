function [ maximum ] = absmax_cu (A, gpuContext)

if nargin < 2
  gpuContext.absmax = absmax_setup (A);
end

kern = gpuContext.absmax.kernel;
out = gpuContext.absmax.outArray;
p2Size = gpuContext.absmax.p2Size;
dataSize = gpuContext.absmax.dataSize;

while p2Size > 1

  [kern, p2Size] = setupKernel (kern, p2Size);
  out = feval (kern, A, dataSize, out);
  A = out;
  
end

maximum = out(1,1);

end


function [kern, gridDim] = setupKernel (kern, dataSize)

blockSize = min (kern.MaxThreadsPerBlock, dataSize);

blockDim = blockSize / 2;
gridDim = dataSize / blockSize;
sharedMem = blockSize * 8;

kern.SharedMemorySize = sharedMem;
kern.GridSize = [gridDim 1];
kern.ThreadBlockSize = [blockDim 1 1];

end
