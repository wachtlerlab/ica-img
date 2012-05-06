function [id] = str2chan (chan, onoff)

if nargin < 2
  onoff = '';
end

if ischar(chan(1))
    chanNum = char2chan(chan(1)); 
    if (length (chan) > 1)
      onoff = chan(2:end);
    end
else
    chanNum = chan;
end

if chanNum < 1 || chanNum > 3
  warning ('ICA:invalid_channel_id', 'Invalid channel id')
end

id = uint8(chanNum);

if ~isempty(onoff) 
  
    if ischar(onoff)
       if strcmpi (onoff, 'on') || strcmpi (onoff, '+') 
            onoff = 1;
       elseif strcmpi (onoff, 'off') || strcmpi (onoff, '-') 
           onoff = 0;
       else
           error ('Unkown onoff input');
       end
    end
    
    id = bitor (id, bin2dec ('00010000'));
    id = bitor (id, bitshift (onoff, 5));
end

end

function [chanNum] = char2chan (ch)
if strcmpi (ch, 'S')
  chanNum = 1;
elseif strcmpi (ch, 'M')
  chanNum = 2;
elseif strcmpi (ch, 'L')
  chanNum = 3;
else
  error ('Unkown input!');
end

end
