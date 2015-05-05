function [ pcs, idx ] = plotDiDirs3D (A)


chmap{1} = 'S+';
chmap{2} = 'S-';
chmap{3} = 'M+';
chmap{4} = 'M-';

N = 4;
A = sortAbf(A);
[~, n] = size (A);
B = reshape (A, N, 7*7, n);

pcs = zeros (N, n);
lts = zeros (N, n);
for k = 1:n
  X = B(:,:,k);
  [pc,~,latent,~] = princomp(X');
  %pc = pc(:,1) * latent (1);
  pcs(:,k) = pc(:,1);
  lts(:,k) = latent;
  %disp (pc);
end

combs = combnk(1:N, 3);

figure();
X = reshape(A, 4, 7*7 * n);
for c = 1:length(combs)
  subplot (2,2,c)
  %combi = combs(c,:);
  x = X(combs(c, 1), :);
  y = X(combs(c, 2), :);
  z = X(combs(c, 3), :);
  scatter3(x, y, z, 50*lts(k), '.k');
  xlabel([chmap{combs(c, 1)} ' (x)'])
  ylabel([chmap{combs(c, 2)} ' (y)'])
  zlabel([chmap{combs(c, 3)} ' (z)'])
  xlim([-0.5 0.5])
  ylim([-0.5 0.5])
  zlim([-0.5 0.5])
end

ncluster = 6;
[idx, C] = kmeans (pcs', ncluster, 'Replicates', 5, 'Distance','city');
colors = lines(ncluster);
%lts = lts/max(lts(:));

fig = figure();
for c = 1:length(combs)
  subplot (2,2,c)
  combi = combs(c,:);
  for k = 1:n
    pc = num2cell (pcs (combi, k));
    [x,y,z] = pc{:};
    [t, p, r] = cart2sph (x,y,z);
    tc = (t/pi)*180;
    phic = (p/pi)*180;
    plot (tc, phic, '.k', 'MarkerSize', 50*lts(k));
    xlim([-190 +190]);
    %plot3(t, p, 1.0,'.');
    hold on;
  end
  
  for k = 1:ncluster
    cc = num2cell (C (k, combi));
    [x,y,z] = cc{:};
    [t, p, r] = cart2sph (x,y,z);
    tc = (t/pi)*180;
    phic = (p/pi)*180;
    plot (tc, phic, '+r', 'Color', colors(k, :));
  end
  
  xlabel(['Theta ||' [chmap{combi}]])
  ylabel('Phi');
end

end