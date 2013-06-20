
inc = {
    
    'ee76509'
    'f0e0c58'
    
    'c8ef28b'
    'a55e992'
    '648dd1c'
    '07b0159'
    
    'cbde32b'
    'e77e27d'};

N = length(inc);
table = cell(N, 1);

for n=1:N
   table{n} = eval(['Act_' inc{n}]); 
end

clear N;
clear inc;
