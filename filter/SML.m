function [ filter ] = SML(cfg)

if nargin < 1 || ~isfield(cfg, 'log')
  log = 0;
else
  log = cfg.log;
end

filter.name = 'SML';
filter.function = @SMLFilterImage;
filter.log = log;

if ~isfield(cfg, 'drift_corr')
  filter.drift_corr = 0;
else
  filter.drift_corr = cfg.drift_corr;
end

if ~isfield(cfg, 'crop')
  filter.crop = 0;
else
  filter.crop = cfg.crop;
end

if ~isfield(cfg, 'scale')
  filter.scale = 0;
else
  filter.scale = cfg.scale;
end

basepath = fileparts(mfilename('fullpath'));
%basepath = '~/Science/ICA/matlab/filter/';

SMHIJL = load(fullfile(basepath, 'sml', 'SMHIJL.dat'));
filter.SMLmx = SMHIJL([1 2 6],:);

filter.channels = [str2chan('S') str2chan('M') str2chan('L')];

end

function [img] = SMLFilterImage (this, img)

hs_data = img.hs_data;

if this.drift_corr
    fprintf('Correcting image drift\n');
    hs_data = correctDriftHS(hs_data);
end

[~, T] = size (hs_data);
edgeN = sqrt (T);

if edgeN ~= round (edgeN)
    error ('Image data to square!');
end

img.edgeN = edgeN;
smlflat = this.SMLmx*hs_data;

if this.log
  % offset not necessary - value 0 should not occur after SML trafo
  smlflat = log (smlflat);
end

img.data  = reshape (smlflat, 3, edgeN, edgeN); %(c,x,y) [f,c,r]

if this.crop
    img.data = img.data(:, 1:this.crop, 1:this.crop);
end

if this.scale > 1
   data = permute(img.data, [3 2 1]);
   data = imgscaledown(data, this.scale);
   img.data = permute(data, [3 2 1]);
end



img.sml = permute (img.data, [3 2 1]); % compat (y,x,c) [f, c, r]

end