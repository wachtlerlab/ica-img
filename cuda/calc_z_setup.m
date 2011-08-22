function [ gpuContext ] = calc_z_setup (A, blockSize)

[gridDim, ~] = size (A);

kern = parallel.gpu.CUDAKernel ('calc_z.ptx', 'calc_z.cu');
kern.GridSize = [1 gridDim];
kern.ThreadBlockSize = [blockSize 1 1];

gpuContext.kern = kern;

end