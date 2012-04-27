function [ t ] = gen_ctime()
t = datestr (clock (), 'yyyymmddHHMM');
end

