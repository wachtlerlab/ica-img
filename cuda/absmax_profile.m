
a = randn(294);

n = 100;

ts = zeros (n,1);
for x=1:n
  tStart = tic;
  y = max (abs(a(:)));
  tEnd = toc (tStart);
  ts(x) = tEnd;
end

mts_cpu = mean (ts);
fprintf ('CPU: %f (%d)\n', mts_cpu, n);

dev = gpuDevice;
gpuContext.absmax = absmax_setup (a);

b = gpuArray (a);

for x=1:n
  tStart = tic;
  y = absmax_cu (b, gpuContext);
  tEnd = toc (tStart);
  ts(x) = tEnd;
end

mts_gpu = mean (ts);

fprintf ('GPU: %f (%d) [%s]\n', mts_gpu, n, dev.Name);
fprintf ('Ratio: %f\n', mts_gpu/mts_cpu);