function [ fh ] = plotActscree(Act, doprint)

if ~exist('doprint', 'var'); doprint == 0; end;

fh = figure('Name','Scree', 'Color', 'w', 'PaperType', 'A4', ...
    'Position', [300, 300, 300, 300]);
hold on;

if ~iscell(Act)
    Act = {Act};
end

N = length(Act);
M = zeros(N, 1);
%colors = jet(N);
if N == 2
        colors = [0.137255 0.392157 0.549020;
            0.858824 0.596078 0.439216];
elseif N == 3
    colors = [
       0.988235 0.603922 0.568627;
       0.431373 0.709804 0.988235;
       0.792157 0.878431 0.474510;
    ];
else
    colors = hsv(N);

end

for n=1:N
  [m, p] = plotScree(Act{n}, colors(n, :));
  M(n) = m;
end

ylim([0, 1])
xlim([0, max(M)]);

if doprint
    set(fh, 'PaperUnits', 'centimeters', 'PaperSize', [8.5 8.5], 'PaperPosition', [0, 0, 8.5, 8.5]);
    print(fh, '-depsc2', '-r300', '-loose', ['scree-plot'])
end

end

function [M, p] = plotScree(Act, color)

M = size(Act.w, 2);
ml = zeros(M, 8);
for imgnr=1:8
    dcsr = std(Act.w(Act.offset(imgnr,1):Act.offset(imgnr,2), :), 1);
    dcsr_n = dcsr/max(dcsr);
    [dcsr_ns, idx] = sort(dcsr_n, 2, 'descend');
    
    ml(:, imgnr) = dcsr_ns;
    %plot(dcsr_ns);
end

x = 1:M;
y = mean(ml, 2)';
plot(x, y, 'Color', color, 'LineWidth', 1.1);

xlabel('BF index (sorted)')
ylabel('standard deviation (normalized)')

K = 130;
p = polyfit(x(1:K)/K, y(1:K), 1);
fprintf('slope: %5.3f, %5.3f\n', p(1), p(2))

end