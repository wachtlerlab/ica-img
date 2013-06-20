function [ dirpath ] = makeActPaper(Act, figset)

if ~exist('figset', 'var'); figset = 0; end;

defaultFontAxis = get(0,'DefaultAxesFontName');
defaultSizeAxis = get(0,'DefaultAxesFontSize');
defaultFontText = get(0,'DefaultTextFontname');
defaultSizeText = get(0,'DefaultTextFontSize');

set(0,'DefaultAxesFontName', 'Helvetica Neue');
set(0,'DefaultAxesFontSize', 12);
set(0,'DefaultTextFontname', 'Helvetica Neue');
set(0,'DefaultTextFontSize', 12);

%%
bf = 116; %11
imgnr = 8; %5

A = AfromActivations(Act);

dirpath = fullfile('paper', A.name(1:7));
mkdir(dirpath);
basepath = fullfile(dirpath, A.name);

% %% %
% [fh_polar, fh_plane] = plotAcogDirs(A);
% %set(fh, 'PaperPositionMode', 'auto');
% print(fh_polar, '-depsc2', '-r300', '-loose', [basepath '-dirs-polar-cog.eps'])
% print(fh_plane, '-depsc2', '-r300', '-loose', [basepath '-dirs-plane-cog.eps'])
% 
% if 1
%  doFinish(defaultFontAxis, defaultSizeAxis, defaultFontText, defaultSizeText);
%  return;
% end
% %% %

[fhL2, fhBeta] = plotAstats(A);
%set(fh, 'PaperPositionMode', 'auto');
set(fhL2, 'PaperUnits', 'centimeters', 'PaperSize', [8.5 8.5], 'PaperPosition', [0, 0, 8.5, 8.5])
print(fhL2, '-depsc2', '-r300', '-loose', [basepath '-stats-l2.eps'])
set(fhBeta, 'PaperUnits', 'centimeters', 'PaperSize', [8.5 8.5], 'PaperPosition', [0, 0, 8.5, 8.5])
print(fhBeta, '-depsc2', '-r300', '-loose', [basepath '-stats-beta.eps'])

fh = plotABetaHist(A);
set(fh, 'PaperUnits', 'centimeters', 'PaperSize', [8.5 8.5], 'PaperPosition', [0, 0, 8.5, 8.5])
print(fh, '-depsc2', '-r300', '-loose', [basepath '-stats-beta-hist.eps'])

fh = plotABeta(A, 1);
set(fh, 'PaperUnits', 'centimeters', 'PaperSize', [8.5 8.5], 'PaperPosition', [0, 0, 8.5, 8.5])
print(fh, '-depsc2', '-r300', '-loose', [basepath '-stats-beta-sorted.eps'])

fh = plotAbetaVsL2(A);
set(fh, 'PaperUnits', 'centimeters', 'PaperSize', [8.5 8.5], 'PaperPosition', [0, 0, 8.5, 8.5])
print(fh, '-depsc2', '-r300', '-loose', [basepath '-stats-beta-norm.eps'])


fh = plotAKurt(A);
set(fh, 'PaperUnits', 'centimeters', 'PaperSize', [8.5 8.5], 'PaperPosition', [0, 0, 8.5, 8.5])
print(fh, '-depsc2', '-r300', '-loose', [basepath '-stats-kurt.eps'])


fh = plotAL2vsKurt(A);
set(fh, 'PaperUnits', 'centimeters', 'PaperSize', [8.5 8.5], 'PaperPosition', [0, 0, 8.5, 8.5])
print(fh, '-depsc2', '-r300', '-loose', [basepath '-stats-kurt-norm.eps'])


fh = plotBfs(A);
%set(fh, 'PaperPositionMode', 'auto');
print(fh, '-depsc2', '-r300', '-loose', [basepath '-ata.eps'])

fh = plotAChroma(A);
%set(fh, 'PaperPositionMode', 'auto');
print(fh, '-depsc2', '-r300', '-loose', [basepath '-chroma.eps'])

[fh_polar, fh_plane] = plotADirs(A);
%set(fh, 'PaperPositionMode', 'auto');
print(fh_polar, '-depsc2', '-r300', '-loose', [basepath '-dirs-polar.eps'])
print(fh_plane, '-depsc2', '-r300', '-loose', [basepath '-dirs-plane.eps'])

[fh_polar, fh_plane] = plotADirs(A, 1);
%set(fh, 'PaperPositionMode', 'auto');
print(fh_polar, '-depsc2', '-r300', '-loose', [basepath '-dirs-polar-eq.eps'])
print(fh_plane, '-depsc2', '-r300', '-loose', [basepath '-dirs-plane-eq.eps'])


[fh_polar, fh_plane] = plotAcogDirs(A);
%set(fh, 'PaperPositionMode', 'auto');
print(fh_polar, '-depsc2', '-r300', '-loose', [basepath '-dirs-polar-cog.eps'])
print(fh_plane, '-depsc2', '-r300', '-loose', [basepath '-dirs-plane-cog.eps'])


[fhA, fhB] = plotAspace(A);
%set(fh, 'PaperPositionMode', 'auto');
set(fhA, 'PaperUnits', 'centimeters', 'PaperSize', [8.5 8.5], 'PaperPosition', [0, 0, 8.5, 8.5])
print(fhA, '-depsc2', '-r300', '-loose', [basepath '-space-dir.eps'])

set(fhB, 'PaperUnits', 'centimeters', 'PaperSize', [8.5 8.5], 'PaperPosition', [0, 0, 8.5, 8.5])
print(fhB, '-depsc2', '-r300', '-loose', [basepath '-space-cog.eps'])

if figset == 0
    doFinish(defaultFontAxis, defaultSizeAxis, defaultFontText, defaultSizeText);
    return;
end

%%
fh = plotACoeffStats(A, Act);
print(fh, '-depsc2', '-r300', '-loose', [basepath '-coeff.eps'])


fh = plotActSingleFit(Act, bf, imgnr);
figname = sprintf('%s-fit-%d-%d.eps', basepath, bf, imgnr);
set(fh, 'PaperUnits', 'centimeters', 'PaperSize', [4.25 4.25], 'PaperPosition', [0, 0, 4.25, 4.25])
print(fh, '-depsc2', '-r300', '-loose', figname)

fh = plotActSingleImg(Act, bf, imgnr);
figname = sprintf('%s-img-%d-%d.eps', basepath, bf, imgnr);
set(fh, 'PaperUnits', 'centimeters', 'PaperSize', [4.25 4.25], 'PaperPosition', [0, 0, 4.25, 4.25])
print(fh, '-depsc2', '-r300', '-loose', figname)

fh = plotActSingleATA(Act, bf, imgnr);
figname = sprintf('%s-ata-%d-%d.eps', basepath, bf, imgnr);
set(fh, 'PaperUnits', 'centimeters', 'PaperSize', [4.25 4.25], 'PaperPosition', [0, 0, 4.25, 4.25])
print(fh, '-depsc2', '-r300', '-loose', figname)


fh = plotAImgData(Act.Model, imgnr);
figname = sprintf('%s-imgdata-%d.eps', basepath, imgnr);
set(fh, 'PaperUnits', 'centimeters', 'PaperSize', [4.25 4.25], 'PaperPosition', [0, 0, 4.25, 4.25])
print(fh, '-depsc2', '-r300', '-loose', figname)

doFinish(defaultFontAxis, defaultSizeAxis, defaultFontText, defaultSizeText);
%%
end

function doFinish(defaultFontAxis, defaultSizeAxis, defaultFontText, defaultSizeText)
%Restore default fonts
set(0,'DefaultAxesFontName', defaultFontAxis);
set(0,'DefaultAxesFontSize', defaultSizeAxis);
set(0,'DefaultTextFontname', defaultFontText);
set(0,'DefaultTextFontSize', defaultSizeText);

close all;

end

