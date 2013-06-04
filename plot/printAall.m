function [ ] = printAall(A)


fh = plotAstats(A);
%set(fh, 'PaperPositionMode', 'auto');
print(fh, '-depsc2', '-r300', '-loose', [A.name '-stats.eps'])

fh = plotBfs(A);
%set(fh, 'PaperPositionMode', 'auto');
print(fh, '-depsc2', '-r300', '-loose', [A.name '-ata.eps'])

fh = plotAChroma(A);
%set(fh, 'PaperPositionMode', 'auto');
print(fh, '-depsc2', '-r300', '-loose', [A.name '-chroma.eps'])

[fh_polar, fh_plane] = plotADirs(A);
%set(fh, 'PaperPositionMode', 'auto');
print(fh_polar, '-depsc2', '-r300', '-loose', [A.name '-dirs-polar.eps'])
print(fh_plane, '-depsc2', '-r300', '-loose', [A.name '-dirs-plane.eps'])

fh = plotAspace(A);
%set(fh, 'PaperPositionMode', 'auto');
print(fh, '-depsc2', '-r300', '-loose', [A.name '-space.eps'])

end

