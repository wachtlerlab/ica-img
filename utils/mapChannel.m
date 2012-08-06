function [channels] = mapChannel (chanlist)

N = length (chanlist);
channels = zeros (1, N*2);

for n = 1:N
  ch = chanlist(n);
  channels(1, (n*2-1)) = str2chan (ch, 'on');
  channels(1, n*2) = str2chan (ch, 'off');
end

end