function [ gpuContext ] = absmax_setup (A)

k_absmax = parallel.gpu.CUDAKernel ('absmax_kernel.ptx',...
                                    'absmax_kernel.cu');                                  
gpuContext.kernel = k_absmax;
dataSize = length (A)*2;
gpuContext.dataSize = dataSize;
p2Size = pow2 (nextpow2 (dataSize));
gpuContext.p2Size = p2Size;
outSize = min (k_absmax.MaxThreadsPerBlock, p2Size) * 8;
gpuContext.outArray = parallel.gpu.GPUArray.zeros (1, outSize);

end

