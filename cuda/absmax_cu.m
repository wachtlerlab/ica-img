function [ maximum ] = absmax_cu (A, kern)

s = size (A);
l = s(1)*s(2);

p2 = pow2 (nextpow2 (l));

A_flat = reshape (A, 1, l);
A_fill = parallel.gpu.GPUArray.zeros (1, p2 - l);
A = cat (2, A_flat, A_fill);

if nargin < 2
  kern = parallel.gpu.CUDAKernel ('absmax_kernel.ptx', 'absmax_kernel.cu');
end

[gridDim, ~] = calcKernParams (kern, p2);
out = parallel.gpu.GPUArray.zeros (1, gridDim);

dataSize = p2;

while dataSize > 1

  [kern, dataSize] = setupKernel (kern, dataSize);
  out = feval (kern, A, out);
  A = out;
  
end


maximum = out(1,1);

end


function [kern, gridDim] = setupKernel (kern, dataSize)

[gridDim, blockDim, sharedMem] =  calcKernParams (kern, dataSize);

kern.SharedMemorySize = sharedMem;
kern.GridSize = [gridDim 1];
kern.ThreadBlockSize = [blockDim 1 1];


end

function [gridDim, threads, sharedMem] = calcKernParams (kern, dataSize)

blockSize = min (kern.MaxThreadsPerBlock, dataSize);

threads = blockSize / 2;
gridDim = dataSize / blockSize;
sharedMem = blockSize * 8;

end