function [filePath] = genReportFigures (Model)

Model = sortModelA (Model);
nick = Model.id(1:7);

reportDir = fullfile ('..', 'reports');

if exist (reportDir, 'dir') == 0
   mkdir (reportDir); 
end

filePath = fullfile (reportDir, [Model.name '-' nick '.ps']);

if exist (filePath, 'file') ~= 0
    delete (filePath);
end

fprintf ('Generating figures for %s (%s)\n\t [saveing to %s]\n',...
  Model.name, Model.id(1:7), filePath);

[~,M] = size (Model.A);

if Model.dataDim == 6
  for n = 1:2
    Mx = CSRecToLMS (Model, n);
    plotBfAndAxis (Mx, filePath);
    plotDirections (Mx, filePath);
  end
  
  plotDirections (Model, filePath);
  
elseif Model.dataDim == 4
   splitAndPlotFN (Model, @plotRatios, 50, filePath);
else
   plotBfAndAxis (Model, filePath);
   plotDirections (Model, filePath);
end

stepSize = 20;
r = rem (M, stepSize);

for start = 1:stepSize:(M - r)
  ppBfSingle (Model, start, stepSize, filePath);
end

ppBfSingle (Model, M-r, r, filePath);

end


function [hf] = genFig ()
  hf = figure ('Position', [0, 0, 800, 1000], 'Color', 'w', 'PaperType', 'A4');
end


function splitAndPlotFN (Model, plotFN, stepSize, filePath)

[~,M] = size (Model.A);
r = rem (M, stepSize);

for start = 1:stepSize:(M - r)
  feval (plotFN, Model, start:start+stepSize, filePath);
end

feval (plotFN, Model, M-r:M, filePath);

end


function plotDirections (Model, filePath)
hf = genFig ();
plotAxisDirs (Model, [], hf);
do_print (hf, filePath, '600');
close (hf);
end


function plotRatios (Model, range, filePath)
hf = genFig ();
plotRatio (Model, range, hf);
do_print (hf, filePath, '600');
close (hf);
end


function plotBfAndAxis (Mx, filePath)
  hf = genFig ();
  plotAbf (Mx, hf);
  do_print (hf, filePath);
  close (hf);
  
  hf = genFig ();
  plotAxis (Mx, hf);
  do_print (hf, filePath);
  close (hf);
end


function ppBfSingle (Model, start, num, filePath)
  hf = genFig ();
  plotABfCSRect (Model, start, num, hf);
  do_print (hf, filePath);
  close (hf);
end


function do_print (hf, filePath, res)

if nargin < 3
  res = '300';
end

  print (hf, '-dpsc2', '-append', ['-r' res], '-zbuffer', filePath);
  
end
