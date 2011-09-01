function [hf] = plotAxisDirsFlat (Model, range, fh)

name = Model.name;
Model = sortModelA (Model);

if nargin < 3
  hf = figure ('Name', ['Directions of the Basis Functions ', name]);
else
  set (0, 'CurrentFigure', fh);
  hf = fh;
end

[~,M] = size(Model.A);

% on channel
if isfield (Model, 'dataDim') && Model.dataDim == 6
  On = getChannels (Model, [1 3 5]);
  Off = getChannels (Model, [2 4 6]);
  A = cat (2, On, Off);
  num = 2*M;
else
  A = Model.A;
  num = M;
end

B = reshapeAbf (A, 0);

if nargin < 2 || isempty (range)
  range = 1:num;
end

pcs = zeros (2, length (range));
for idx=range
    slice = squeeze (B(idx,:,:,:));
    slice = reshape (slice, 49, 3);
    pc = getPC (slice);
    pcs(:,idx) = pc;
end

%pcs(1,:) -> X; pcs(2,:) -> Y
%x = [pcs(1,:); pcs(1,:)];
%y = [zeros(1,length(pcs(2,:))); abs(pcs(2,:))];

tt = atan2(pcs(2,:), pcs(1,:));
rr = sqrt (pcs(1,:).^2 + pcs(2,:).^2);

tt = cat (2, tt, tt + pi); %mirror directions
rr = cat (2, rr, rr);

tt = tt * 180 / pi;

x = [tt; tt];

%ft = do_gauss (-10:0.1:10,0,2)*10;
%rr = conv (rr, ft, 'same');

y = [zeros(1,length(rr)); rr];

hold on
line (x, y, 'Color', 'black');
xlim([0 180])

end

function [pc] = getPC (basisFunction)


slice = 0.5 + 0.5 * basisFunction / max(abs(basisFunction(:)));

S = squeeze (slice(:,1));
M = squeeze (slice(:,2));
L = squeeze (slice(:,3));

x = L-M;
y = S-((L+M)/2);

X = [x'; y']';

[pc,~,latent,~] = princomp(X);
pc = pc(:,1) * latent (1);

end


function [ shapedA ] = reshapeAbf (A, doflip)
%reshapeAbf Reshape the mixing matrix A to obtain the basis functions

if isa (A, 'struct')
    A = A.A;
end

if ndims (A) ~= 2
   error ('Matrix must be 2D')
end

if nargin < 2
    doflip = 1;
end

[m, n] = size (A);

c = sqrt (m/3);

if (c ~= round (c))
    error ('Basis function not like X*X*3')
end

if doflip
    A = flipdim (A, 1);
end

R = reshape (A, 3, c, c, n);
shapedA = permute (R, [4 3 2 1]);

end

function [out] = getChannels (Model, channels)
C = num2cell (Model.A, 1);
selChan = @(X) selectChannel (X, channels);
C = cellfun (selChan, C, 'UniformOutput', false);
out = cell2mat (C);

end

function [out] = selectChannel (Abf, cols)

shaped = reshape (Abf, 6, 7*7);
X = shaped(cols, :);
out = reshape (X, 3*7*7, 1);

end


function [out] = do_gauss (x, mu, sigma)

out = 1/(2*pi*sigma^2)*exp(-(x - mu).^2/(2*sigma^2));

end