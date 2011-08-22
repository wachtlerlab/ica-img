function [ Z ] = calc_z (S, prior, gpuContext)


mu = gpuArray (prior.mu);
beta = gpuArray (prior.beta);
sigma = gpuArray (prior.sigma);

[gridDim, blockDim] = size (S);

if nargin > 2
  kern = gpuContext.calc_z.kern;
else
  kern = parallel.gpu.CUDAKernel ('calc_z.ptx', 'calc_z.cu');
  kern.GridSize = [1 gridDim];
  kern.ThreadBlockSize = [blockDim 1 1];
end

Z = parallel.gpu.GPUArray.zeros (gridDim, blockDim);
Z = feval (kern, S, mu, beta, sigma, Z);

end

