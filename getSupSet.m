
inc = {
    'b553623'
    '5bfa23b'
    'f0e0c58'
    '5136603'
    'e82a877'
    '37ba076'
    'c8ef28b'
    'a55e992'
    '648dd1c'
    '07b0159'
    '280a0e3'
    'a874ece'
    'e77e27d'
    '4aeedfa'
    '6a0ba71'
    'ee76509'
    'cbde32b'};

N = length(inc);
table = cell(N, 1);

for n=1:N
   table{n} = eval(['Act_' inc{n}]); 
end

clear N;
clear inc;
