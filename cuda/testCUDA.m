cuda = CUDA()
cuda.setup()

m = 100;
k = 200;
n = 300;

A = randn(m,k);
B = randn(k,n);
C = randn(m,n);

%im = cuda.idamax(A);

cuda.gemm(C, A, B, 1, 0);

X = B'*A';
%clear ('cuda')
%unloadlibrary('icudamat');