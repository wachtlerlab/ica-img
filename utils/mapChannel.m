function [channels] = mapChannel (chanlist, rectify)

if nargin < 2
    rectify = 1;
end

N = length (chanlist);

if rectify
    M = N*2; 
else
    M = N;
end

channels = zeros (1, M);

for n = 1:N
  ch = chanlist(n);
  if rectify
    channels(1, (n*2-1)) = str2chan (ch, 'on');
    channels(1, n*2) = str2chan (ch, 'off');
  else
     channels(1, n) = str2chan (ch); 
  end
end

end