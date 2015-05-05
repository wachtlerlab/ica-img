function plotShist(S,nbins,layout)
% plotShist -- 
%   Usage
%     plotShist(S [, nbins])
%   Inputs
%      S         the fitted basis coefficients
%      nbins     number of bins to use for S density plot
%      layout    optional argument to specify [nrows, ncolumns]
%   Outputs
%
% Michael Lewicki 9/97
% lewicki@salk.edu

[M,N] = size(S);

if (exist('nbins','var') == 0)
  nbins = min(51,ceil(N/25));
end
if (exist('layout','var') == 0 | isempty(layout))
  layout(2) = ceil(sqrt(M));
  layout(1) = ceil(M/layout(2));
end

nrows = layout(1);
ncols = layout(2);

uselog = 0;
showpeak = 1;
peakrange = 99.9;
usefixedrange = 1;
fixedmaxs = 5;

cliptails = 1;		% prevent tails from increasing binsize
tailrange = 99.9;		

if (nrows*ncols < M)
  % select sample of coeffs
  r = ceil(M*rand(nrows*ncols,1));
else
  r = 1:M;
end

pidx=0;
for m=1:length(r);
  pidx = pidx + 1;
  m = r(pidx);
  if (M == 1)
    s = S;
  else
    s = S(m,:);
    subplot(nrows,ncols,pidx);
  end

  mins=min(s);
  maxs=max(s);

  if cliptails
    mins = prctile(S(:),(100 - tailrange)/2);
    maxs = prctile(S(:),100 - (100 - tailrange)/2);
    % make symetric around zero
    maxs = 0.5*(abs(mins) + abs(maxs));

    % fprintf('Clipping S outside [%f,%f]\n',maxs,maxs);
    hpdidx = find(s > -maxs & s < maxs);
    s = s(hpdidx);
  end

  if (uselog == 0)
    hist(s,nbins);

  else
    if usefixedrange
      maxs = fixedmaxs;
    end

    % nbins should be even to get bin with center at zero
    ds = 2*maxs/(nbins-1);
    X = -maxs:ds:maxs;
    N = hist(s,X);

    norm = sum(N);
    P = N/norm;
    sumP = sum(P);
  
    % fprintf('sum(P) = %f\n',sumP);
    % fprintf('binsize = %f\n',ds);

    % reset(gcf);
    zidx = find(P < 1/norm);
    P(zidx) = 1/norm;
    semilogy(X,P);

  end

  if showpeak
    mins = prctile(s,(100 - peakrange)/2);
    maxs = prctile(s,100 - (100 - peakrange)/2);
    set(gca, 'XLim',[mins,maxs]);
  else
    axis tight;
  end

  if (M == 1)
    xlabel('S');
    ylabel('P(S)');
    title('Distribution of basis coefficients');
  else
    % xlabel(sprintf('s_{%d}',m));
    ylabel(sprintf('P(s_{%d})',m));
  end
end
