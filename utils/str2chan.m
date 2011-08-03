function [id] = str2chan (chan, onoff)

if ischar(chan)
    
    if strcmpi (chan, 'S')
        chanNum = 1;
    elseif strcmpi (chan, 'M')
        chanNum = 2;
    elseif strcmpi (chan, 'L')
        chanNum = 3;
    else
        error ('Unkown input!');
    end
else
    chanNum = chan;
end

id = chanNum;

if nargin > 1
    
    if ischar(onoff)
       if strcmpi (onoff, 'on')
            onoff = 1;
       elseif strcmpi (onoff, 'off')
           onoff = 0;
       else
           error ('Unkown onoff input');
       end
    end
    
    id = bitor (id, bin2dec ('00010000'));
    id = bitor (id, bitshift (onoff, 5));
end

end
