function [caption] = chan2str (id, mapping)

if nargin > 1
    id = mapping(id);
end
if id == 0
    caption = '';
    return;
end

channelMap = {{'S'}, {'M'}, {'L'}};
onoffMap = {{'-'}, {'+'}};

chan = bitand (id, bin2dec('00001111'));
caption = char (channelMap{chan});

if  bitand (bin2dec('00010000'), id) ~= 0
    idx = (bitand (bin2dec('00100000'), id) ~= 0) + 1;
    onoffstr = char (onoffMap{idx});
    caption = char ([caption onoffstr]);
end

end
