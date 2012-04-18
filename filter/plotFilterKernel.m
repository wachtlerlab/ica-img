function [ output_args ] = plotFilterKernel(kernel)
%PLOTFILTERKERNEL Summary of this function goes here
%   Detailed explanation goes here

nvals = size (kernel, 1);
n = floor (nvals/2.0);

[X, Y] = meshgrid(-n:n);
Z = kernel;

n = floor (length(X)/2)+1;
figure ('Name', 'Kernel', 'Position', [0, 0, 1200, 300]);
subplot(1,3,1);
imshow(Z);
subplot(1,3,2);
hold on;
plot(X(n,:), Z(n,:));
plot(X(n,:), Z(n,:), 'r+');
subplot(1,3,3);
mesh (X, Y, Z);
center = Z(n,n);
Z(n,n) = 0;
fprintf ('sum: %f; ratio: %f\n', sum(Z(:)), center/sum(Z(:)));


end

