function [ outfile ] = genReport (Model)

modelId = Model.name;

tmpFolder = tempname;
mkdir (tmpFolder);
fprintf ('TempFolder: %s\n', tmpFolder);

savePath = path;
path ('config', path);
path ('report', path);

% Results
%opts.format = 'pdf';
%opts.outputDir = tmpFolder;
%opts.showCode = false;
%opts.useNewFigure = false;
%publish ('report_template', opts);

bfh1 = plotBfs (Model, 1);
bfh2 = plotBfs (Model, 140);
bfh3 = plotBfs (Model, 284);

% Model Configuration
publishOptions.format = 'pdf';
publishOptions.outputDir = tmpFolder;
%publishOptions.figureSnapMethod = 'print';
publishOptions.useNewFigure = false;
publishOptions.showCode = true;
publishOptions.evalCode = false;
publish (modelId, publishOptions);


% concat pdf files
curDir = pwd;
cd (tmpFolder);

export_fig ('bf1.pdf', '-pdf', bfh1)
export_fig ('bf2.pdf', '-pdf', bfh2)
export_fig ('bf3.pdf', '-pdf', bfh3)

%pcmd = sprintf ('print -f%d -dpdf bf1.pdf', bfh1);
%eval (pcmd);

%'pdftk %s %s cat output out.pdf', ...
cmd = sprintf ('gs -dNOPAUSE -sDEVICE=pdfwrite -sOUTPUTFILE=out.pdf -dBATCH %s %s > /dev/null', ...
               [modelId '.pdf'], 'bf1.pdf bf2.pdf bf3.pdf');
system (cmd);
cd (curDir);

% copy result to output folder
sourceFile = fullfile (tmpFolder, 'out.pdf');
copyfile (sourceFile, [modelId '.pdf']);

% all done, cleanup
path (savePath);

rmdir (tmpFolder, 's');

outfile = [modelId '.pdf'];

end

