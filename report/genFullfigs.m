function genFullfigs (Model)

Model = sortModelA (Model);
filePath = fullfile ('..', 'results', [Model.name '.ps']);

if exist (filePath, 'file') ~= 0
    delete (filePath);
end

fprintf ('Generating figures for %s (%s)\n\t [saveing to %s]\n',...
  Model.name, Model.id(1:7), filePath);

if Model.dataDim == 6
  for n = 1:4
    Mx = CSRecToLMS (Model, n);
    plotBfAndAxis (Mx, filePath);
  end
elseif Model.dataDim == 3
  plotBfAndAxis (Model, filePath);
else
  fprintf ('Model.dataDim unsupported. No Color/Axis prints.\n');
end

[~,M] = size (Model.A);
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

function do_print (hf, filePath)
  print(hf, '-dpsc2', '-append', filePath);
end
