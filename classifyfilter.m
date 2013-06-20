function [ fklass ] = classifyfilter(filter)

fklass = '';

if isfield(filter, 'surround')
    
    if isstruct(filter.surround)
        surround = mapSurround(filter.surround, filter.center);
         
        for ch = filter.center  
            cx = str2chan(ch);
           chans = surround{cx};
           [surr, w] = surroundGetChannel(chans);
           surrX = arrayfun(@(x) chan2str(x), surr);
           surr_str = zeros(1, length(surrX));
           surr_str(1,:) = surrX;
           fklass = [fklass chan2str(cx) ':{'];
           fklass = [fklass surr_str];
           fklass = [fklass '} '];
        end
        
    else
        fklass = [fklass 'S: ' filter.surround];
    end
end

if isfield(filter, 'log')
    fklass = [ fklass '; L: ' num2str(filter.log)];
end

end

function [surround] = mapSurround(surround, center)

if isstruct(surround)
    names = fieldnames(surround);
    idx = cell(length(names), 1);
    for n = 1:length(names)
        ch = char(names(n));
        val = surround.(ch);
        
        if isstruct(val)
            keys = fieldnames(val);
            weights = cell(3, 1);
            for ki = 1:length(keys)
                surr_ch = char(keys(ki));
                surr_idx = str2chan(surr_ch);
                weights{surr_idx} = val.(surr_ch);
            end
            idx{str2chan(ch)} = weights;
        else
            idx{str2chan(ch)} = chanlist2idx(val);
        end
        
    end
    surround = idx;
else
    S = cell(length(center), 1);
    for n = 1:length(center)
        ch = center(n);
        S{str2chan(ch)} = chanlist2idx(surround);
    end
    surround = S;
end

end

function [channels, weights] = surroundGetChannel(channels)

if isempty(channels)
    channels = [];
    weights = [];
    return;
end

if iscell(channels)
    % cant use cell2mat since that ignores leading zeros
    new_channels = zeros(length(channels),1);
    for idx=1:length(channels)
        new_channels(idx) = channels{idx};
    end
    channels = new_channels;
    valid_channels = channels ~= 0;
    pos = find(valid_channels);
    weights = channels(pos);
    channels = pos;
else
    nch = length(channels);
    weights = ones(nch, 1) / nch;
end

end