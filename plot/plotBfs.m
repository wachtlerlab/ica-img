function [fig, h] = plotBfs (A, h, landscape)

if ~exist('landscape','var'); landscape = 0; end;

buf   = 1;
[L, M] = size(A.rgb);
sz = sqrt(L/A.nchan);

[m,n] = plotCreateGrid(M, landscape);

array = ones(buf+m*(sz+buf),buf+n*(sz+buf), A.nchan);

k=1;
for i=1:m
  for j=1:n
      if (k > M)
          break;
      end
    bf   = reshape(A.rgb(:,k), A.nchan, sz, sz);
    bf   = permute (bf, [3 2 1]); 
    array(buf+(i-1)*(sz+buf)+[1:sz],buf+(j-1)*(sz+buf)+[1:sz],:) = bf;
    k=k+1;
  end
end

if exist('h','var') && ~isempty(h)
  set (h,'CData',array);
else
%   fs = [900, 600];
%   if landscape
%       fs = sort(fs);
%   end
  
 % fig = figure ('Name', ['Basis Functions: ', A.name], ...
 %   'Position', horzcat([0, 0], fs), 'Color', 'w', 'PaperType', 'A4');

 fig = plotACreateFig(A, 'Basis Functions', landscape, [900, 600]);
 h = imagesc (array, 'EraseMode', 'none',  [-1 1]);
 title(['Basis Functions  ' A.name]);
 
 axis tight image off
 set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
end


end

