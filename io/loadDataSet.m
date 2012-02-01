function [ dataset ] = loadDataSet (filename, loc)

path = filename;

if nargin < 2
  loc = '/ds/0';
end

imgdata = h5read (path, [loc '/imgdata']);
indicies = h5read (path, [loc '/indicies']);
patsperm = h5read (path, [loc '/patsperm']);
Ainit    = h5read (path, [loc '/Ainit']);

patchsize = h5readatt (path, loc, 'patchsize');
[npats, nclusters] = size (patsperm);
blocksize = h5readatt (path, loc, 'blocksize');


dataset.imgdata   = imgdata;
dataset.patchsize = uint16 (patchsize);
dataset.npats     = npats;
dataset.blocksize = blocksize;
dataset.nclusters = nclusters;
dataset.indicies  = indicies;
dataset.patsperm  = patsperm;

dataset.Ainit = Ainit;

end

