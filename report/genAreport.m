function [ tmpFolder ] = genAreport(A)

tmpFolder = tempname();
mkdir (tmpFolder);
fprintf ('TempFolder: %s\n', tmpFolder);

publishOptions.format = 'pdf';
publishOptions.outputDir = tmpFolder;
%publishOptions.figureSnapMethod = 'print';
publishOptions.useNewFigure = false;
publishOptions.showCode = false;

publish('reportA', publishOptions)

sourceFile = fullfile (tmpFolder, 'reportA.pdf');
copyfile (sourceFile, [A.name '.pdf']);


end

