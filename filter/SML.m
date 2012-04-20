function [ filter ] = SML(cfg)

if nargin < 1 || ~isfield(cfg, 'log')
  log = 0;
else
  log = cfg.log;
end

filter.name = 'SML';
filter.function = @SMLFilterImage;
filter.log = log;

SMHIJL = load('sml/SMHIJL.dat');
filter.SMLmx = SMHIJL([1 2 6],:);

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

tmp = reshape (smlflat, 3, edgeN, edgeN);
sml = permute (tmp, [3 2 1]);

%compat field
img.SML = sml;
img.data = sml;

end