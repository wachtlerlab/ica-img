function [ filter ] = SML(cfg)

if nargin < 1 || ~isfield(cfg, 'log')
  log = 0;
else
  log = cfg.log;
end

filter.name = 'SML';
filter.function = @SMLFilterImage;
filter.log = log;

basepath = '~/Coding/ICA/matlab/filter/';

SMHIJL = load([ basepath 'sml/SMHIJL.dat']);
filter.SMLmx = SMHIJL([1 2 6],:);

filter.channels = [str2chan('S') str2chan('M') str2chan('L')];

end

function [img] = SMLFilterImage (this, img)

[~, T] = size (img.hs_data);
edgeN = sqrt (T);

if edgeN ~= round (edgeN)
    error ('Image data to square!');
end

img.edgeN = edgeN;
smlflat = this.SMLmx*img.hs_data;

if this.log
  % offset not necessary - value 0 should not occur after SML trafo
  smlflat = log (smlflat);
end

img.data  = reshape (smlflat, 3, edgeN, edgeN); %(c,x,y) [f,c,r]
img.sml = permute (img.data, [3 2 1]); % compat (y,x,c) [f, c, r]

end