function [idx] = chanlist2idx (chanlist)

N = length(chanlist);
idx = zeros (1, N);
for n = 1:length(chanlist)
  ch = chanlist(n);
  idx(n) = str2chan(ch);
end

end