function [ curfig ] = plotActSingleFit(Act, bfnr, imgnr)


wall   = Act.w(:, bfnr);
offset = Act.offset;

pos = [300, 300];
fs = [500, 500];
figtitle = ['Fit ' num2str(bfnr) ' ' num2str(imgnr)];
curfig = figure('Name', figtitle, 'Position', horzcat(pos, fs), ...
    'Color', 'w', 'PaperType', 'A4');
set(0,'CurrentFigure', curfig);

color = [0.121569 0.427451 0.658824];

epp = Act.epp(bfnr, imgnr, :);
w = wall(offset(imgnr,1):offset(imgnr,2), :);

xrange = min(w):0.01:max(w);

mu = epp(1);
sigma = epp(2);
beta = epp(3);

y = expwrpdf(xrange, epp(1), epp(2), epp(3));

fprintf('mu:%f sigma:%f beta:%f\n', mu, sigma, beta);
        
[nelm, xcnt] = hist(w, 40);
[ax, h1, h2] = plotyy(xcnt, nelm, xrange, y, 'bar', 'plot');
hold on;
xlim([-6 6])
box off

yticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
%set(h1,'FaceColor', color,'EdgeColor', color)
set(h1,'FaceColor', 'black','EdgeColor', 'black')
set(h2, 'LineWidth', 1.2, 'Color', 'red'); %[0.752941 0.211765 0.168627]

set(ax(2), 'YTick', yticks)

stwide = 0.01;
intwidth = 6;
x = (mu-intwidth):stwide:(mu+intwidth);
intx = @(b) integral(@(x) expwrpdf(x, mu, sigma, beta), -Inf, b);
yf = arrayfun(intx, x);
set (gcf, 'CurrentAxes', ax(2));
hold on;
plot(x, yf, 'Color', 'blue', 'LineWidth', 1.1); %[0.505882 0.847059 0.815686]
xlim([-6 6])
ylim([0 1.1])


%idx_a = find(yf > (border*0.5), 1);
%idx_b = find(yf > 1 - (border*0.5), 1);


end

