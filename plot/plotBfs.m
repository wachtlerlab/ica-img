function [fig, h] = plotBfs (A, h)

buf   = 1;
[L, M] = size(A.rgb);
sz = sqrt(L/A.nchan);

if floor(sqrt(M))^2 ~= M
  m = ceil(sqrt(M/2));
  n = ceil(M/m);
else
  m = sqrt(M);
  n = m;
end

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

if exist('h','var')
  set (h,'CData',array);
else
  fig = figure ('Name', ['Basis Functions: ', A.name], ...
    'Position', [0, 0, 800, 1000], 'Color', 'w', 'PaperType', 'A4');
  h = imagesc (array, 'EraseMode', 'none',  [-1 1]);
  
  axis tight image off
end


end

