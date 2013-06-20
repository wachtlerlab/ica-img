function [ fig ] = plotATunings(A)

fig = figure('Name', [A.name ' Tunings'], 'Color', 'w', ...
    'Position', [0, 0, 800, 400]);
set(0,'CurrentFigure', fig);

pcs = A.pcs;
%tt = atan2(pcs(2,:), pcs(1,:));
rr = hypot(pcs(1,:), pcs(2,:));
%tt(tt < 0) = tt(tt < 0) + pi; % negative values get positive here

nbf = A.nbf;

angles = zeros(nbf, 1);
%distances = zeros(nbf, 1);
for bfidx = 1:nbf
    [theta, r] = calcATuningSingle(A, bfidx);
    angles(bfidx, :) = theta;
    %distances(bfidx, :) = r;
end

tt = angles';
%rr = distances';

tt = tt * 180 / pi;

g = ones(length(tt), 1);

hh = squeeze(max(A.dkl(:, 3, :))) - squeeze(min(A.dkl(:, 3, :)));
hh = (hh / max(abs(hh(:)))) * 90;
%hh = cat (1, hh, hh);

step = 360/60;
x = 0:step:360;
nx = histc(tt, x);
h = scatterhist(tt, hh, 'Marker', '.', 'NBins', 60, 'Direction','out', ...
    'Location','NorthEast', 'Color','k', 'Legend', 'off');


xlim([0, 360]);
ylim([0, 90]);
xlabel('Hue angle')
ylabel('Elevation')

if 0
hold on;
set(fig, 'CurrentAxes', h(2));
color = 'k';
bar(x+(step/2)+1, nx, 'EdgeColor', color, 'FaceColor', color);
axis tight;
box off;
axis off;

set(fig, 'CurrentAxes', h(3));

x = 0:90/10:90;
y = histc(hh, x);
barh(x, y, 'EdgeColor', color, 'FaceColor', color);
xlim([0, 90])
%set(gca,'CameraUpVector', [1,0,0]);
%set(gca,'CameraPosition', [1 0 0]);
axis tight;
axis off;
box off;
end

end

