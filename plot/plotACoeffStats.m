function [ fig ] = plotACoeffStats(A, Act)

N = 25;

[nrows, ncols] = plotCreateGrid(N);
landscape = 0;

fig = plotACreateFig(A, 'Coeff Stats', landscape, [1200, 800]);
hb = tight_subplot(nrows, ncols, [.01 .01], [.05 .01]);



for idx=1:N
    set (gcf, 'CurrentAxes', hb(idx));
    hold on
    
    %hist(Act.w(:, idx), 40);
    %mu = 0;
    %sigma = 1;
    %beta = A.beta(idx);
    %fprintf('ExPwr: mu=%5.2f  sigma=%5.2f  beta=%+5.2f\n', mu,sigma,beta);
    
    [y, x] = hist(Act.w(:, idx),100);
    plot(x,y, 'k');
    axis tight;
    
    if idx ~= (nrows-1)*ncols+1
        axis off;
    else
        ylabel('p')
        xlabel('source activation');
        set(hb(idx),'XTickLabel','');
        set(hb(idx),'YTickLabel','');
        set(hb(idx),'XTick',[]);
        set(hb(idx),'YTick',[]);
    end
    
    k = A.kurt(idx);
    text(0, max(y)*1.1, num2str(k))
    ylim([0, max(y)*1.2])
  
end

for idx=N+1:length(hb)
    set (gcf, 'CurrentAxes', hb(idx));
    axis image off;
end

end
