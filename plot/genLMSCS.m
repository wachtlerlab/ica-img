function genLMSCS (resolution, axis_handle)

if nargin < 1
   resolution = 1/16; 
end

if nargin < 2
    figure ('Name', 'LMS Colorspace')
else
    set (gcf, 'CurrentAxes', axis_handle);
end

data = genColorCombinations (resolution);
plotPatch (data);

plot ([-0.6 0.6], [0 0], '--', 'color', [0.7 0.7 0.7])
text (0.7, 0, 'L-M')

plot ([0 0], [-0.6 0.6], '--', 'color', [0.7 0.7 0.7])
text (0, 0.6, 'S')

end

