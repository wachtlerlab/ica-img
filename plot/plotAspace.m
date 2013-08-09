function [ figA, figB ] = plotAspace(A)


[L, M] = size(A.rgb);
dkl = A.dkl;
pos = zeros(M, 4);

figA = plotACreateFig(A, 'A in DKL [lines]', 0, [500, 500]);
hold on;
line([0, 0], [-1 1], 'Color', [.8 .8 .8]); 
line([-1, 1], [0 0], 'Color', [.8 .8 .8]);

figB = plotACreateFig(A, 'A in DKL [cog]', 0, [500, 500]);
hold on;
line([0, 0], [-1 1], 'Color', [.8 .8 .8]); 
line([-1, 1], [0 0], 'Color', [.8 .8 .8]);


cog = zeros(M, 2);

for idx=1:M
    pcs = A.pcs(:, idx)*10; % was: 15
         
    x = dkl(:, 1, idx);
    y = dkl(:, 2, idx);
    z = dkl(:, 3, idx);
    
    mx = mean(x);
    my = mean(y);

    n = norm(A.sorted(:, idx));
    
    x0 = mx + pcs(1)/2;
    x1 = mx - pcs(1)/2;
    
    y0 = my + pcs(2)/2;
    y1 = my - pcs(2)/2;
    
    pos(idx, :) = [x0 x1 y0 y1];
    
    c = mean(reshape(A.rgb(:,idx), 3, A.ps*A.ps), 2)';
    psize = 1.0*log10(n*25);

    set(0,'CurrentFigure',figA)
    line([x0, x1], [y0, y1], 'LineWidth', psize, 'Color', c);
    
    set(0,'CurrentFigure',figB)
    scatter(mx, my, psize*5, c, 'fill'); % was 20
    cog(idx, :) = [mx, my];
end

icalms476lum1 = [0.315722, 0.684278, 2.8265];
icalms576lum1 = [0.527134, 0.472866, 0.000395835];

icadkl476lum1 = lms2dkl(icalms476lum1);
icadkl576lum1 = lms2dkl(icalms576lum1);

x = [icadkl476lum1(1), icadkl576lum1(1)];
y = [icadkl476lum1(2), icadkl576lum1(2)];

plot(x, y, 'k');

l = 1.15*max(max(abs(pos)));
set_layout(figA, l);
set_layout(figB, l);


mcog = mean(cog);
ccog = cog - repmat(mcog, M, 1);

[pc,~,latent,~] = princomp(ccog);
pc = pc(:,1) * latent (1);
tt = atan2(pc(2,:), pc(1,:));

ttx = atan2(y(1) - y(2), x(1) - x(2));
ttdeg = tt * 180/pi;
ttxdeg = ttx * 180/pi;
fprintf ('angle: %f [%f] - [%f]\n', ttdeg, ttdeg - 90, ttxdeg);

cpc = pc'+mcog;
hold on; plot([cpc(1)*100 mcog(1) -cpc(1)*100], [ cpc(2)*100  mcog(2) -cpc(2)*100], ':k')

end

function set_layout(h, l)
figure(h)

axis equal
xlim([-l l])
ylim([-l l])



end

function [dkl] = lms2dkl(lms)

 L =  lms(:, 1);
 M =  lms(:, 2);
 S =  lms(:, 3);
 
 x = L-M;
 y = S-((L+M)/2);
 z = L+M;
 dkl = [x,y,z];

end