function [ imgact ] = plotActivations(Model, ds, fnr, imgnr)

A = Model.A;
W = pinv(A);

bf = W(fnr,:); %a row of W is the filter 

img = permute(ds.imgdata(:,:,:,imgnr), [3 2 1]);
patches = imgallpatches(img, ds.patchsize);
[~, k] = size(patches);
imgact = zeros(k, 1);

for n=1:k
    patch = patches(:,n);
    act = dot(patch,bf);
    imgact(n) = act;
end

N = sqrt(k);
imgact = reshape(imgact, N, N);

end

