function [ fh ] = plotEstDirft(shifts)

fh1 = figure;
fh2 = figure;
[~, nplanes, nimages] = size(shifts);

colors = jet(nplanes);
[r, c] = plotCreateGrid(nimages);

for n = 1:nimages
    figure(fh1)
    subplot(r, c, n)
    hold on
    scatter(shifts(1,:, n), shifts(2,:, n), 15, colors, 'filled')
    %scatter(shifts(1,:, n), shifts(2,:, n), '+k')
    xlim([-3, 3])
    ylim([-3, 3])
    xlabel('x shift')
    ylabel('y shift')
    
    figure(fh2);
    subplot(nimages, 1, n)
    hold on
    plot(shifts(1, :, n), 'r')
    plot(shifts(2, :, n), 'g')
    ylim([-3, 3])
    legend({'xshift', 'yshift'})
    
end


end

