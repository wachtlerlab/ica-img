function [hf] = plotComps (Model, range, fh)

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

pcs_m = pcs .* -1;
foo = cat (2, pcs, pcs_m);

polar (0, 0.2); % Scaling of the polar plot to the maximum value (FIXME)
hold on

tt = atan2(foo(2,:), foo(1,:));
rr = sqrt (foo(1,:).^2 + foo(2,:).^2);

for idx = 1:length (foo)
   
   t = tt (idx);
   r = rr (idx);
   
   polar ([0 t], [0 r], '-k');
   hold on    
end

end

function [pc] = getPC (basisFunction)


slice = 0.5 + 0.5 * basisFunction / max(abs(basisFunction(:))); 

S = squeeze (slice(:,1));
M = squeeze (slice(:,2));
L = squeeze (slice(:,3));

x = L-M;
y = S-((L+M)/2);

X = [x'; y']';

[pc,score,latent,tsquare] = princomp(X);
pc = pc(:,1) * latent (1);

end

function plotComp (basisFunction, hf)

%colord = 0.5 + 0.5 * basisFunction / max(abs(basisFunction(:))); 
%colord = flipdim (colord, 2);

slice = 0.5 + 0.5 * basisFunction / max(abs(basisFunction(:))); 

S = squeeze (slice(:,1));
M = squeeze (slice(:,2));
L = squeeze (slice(:,3));

x = L-M;
y = S-((L+M)/2);

X = [y'; x']';

[pc,score,latent,tsquare] = princomp(X);
p1 = pc(:,1) * latent (1);
latent (1)

if latent(1) <  0.01
  return;
end

compass (pc(1,1), pc(2,1));
hold on

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