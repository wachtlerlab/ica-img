function printFig(fig, name, x, y, margin)

if ~exist('margin', 'var'); margin = 0; end;

set(fig, 'PaperUnits', 'centimeters', 'PaperSize', ...
    [x y], 'PaperPosition', [0, 0, x+margin(3), y+margin(4)])

print(fig, '-depsc2', '-r300', '-loose', [name '.eps'])

end

