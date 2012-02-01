function [Model] = startICA (modelId, varargin)

if nargin < 1
  modelId = 'color_cs_rect_1';
end;

options = struct('autosave', 1, ...
                 'savestate', 1, ...
                 'progress', 0, ...
                 'gpu', 1);
               
if nargin > 1
  options = parse_varargs (options, varargin);
end

if exist (fullfile ('config', [modelId '.m']), 'file') == 0
  error ('Cannot find config');
end

currev = getCurRev ();

fprintf ('Starting simulation for %s [code: %s]', modelId, currev);

%%
% basic init
clear Model fitPar dispPar Result;

stateDir = fullfile ('..', 'state');

if exist (stateDir, 'dir') == 0
   mkdir (stateDir); 
end


[Model, fitPar, dispPar, dataPar] = loadConfig (modelId);

Model.id = DataHash (Model, struct ('Method', 'SHA-1'));

if nargin > 1
  dispPar.plotflag = options.progress;
  fitPar.saveflag = options.savestate;
end

if dispPar.plotflag
  figure(1)
  figure(2)
  figure(3)
end


fprintf ('\nFitting %s for config %s [%s]\n',...
  Model.id(1:7), Model.cfgId(1:7), datestr (clock (), 'yyyymmddHHMM'));


%% Prepare image data
images = prepare_images (dataPar);

% Present the filtered pictures (inkluding the excluded patches)
% to the user for visual validation
if dispPar.plotflag && dataPar.doDebug
    displayImages (images, dataPar, 1);
end

%% Generate the DataSet
tstart = tic;
fprintf('\nGenerating dataset...\n');
dataset = generateDataSet (images, fitPar, dataPar);
fprintf('   done in %f\n', toc(tstart));


%% Setup the Model & Result structs
%

Result.priorN = 0;
Result.dataIdx = 1;
Result.X = [];		% force new dataset to be generated

Result.iter = 1;
Result.tStart = tic;

Result.S = zeros (length (Model.A), dataset.blocksize);



%% Infer the Model

[Model, Result] = fitModel (Model, fitPar, dispPar, dataset, Result, options);

%% 

% time reporting
Result.tDuration = toc (Result.tStart);

Model.fitPar = fitPar;
Model.dispPar = dispPar;
Model.dataPar = dataPar;
Model.onGPU = options.gpu;
Model.dataset = dataset;

fprintf (['Total time: (',num2str(Result.tDuration),')\n']);
Model.Result = Result;
Model.codeVersion = currev;

if options.autosave
  filename = saveResult (Model);
  fprintf ('Saved results to %s\n', filename);
end

end


function [options] = parse_varargs(options, varargin)

args = size (varargin{1});
for cur = 1:2:args(2)
  opt = char (varargin{1}(cur));
  arg = char (varargin{1}(cur + 1));
  
  switch opt
    case 'autosave'
      options.autosave = str2num (arg);
    case 'savestate'
      options.savestate = str2num (arg);
    case 'progress'
      options.progress = str2num (arg);
    case 'gpu'
      options.gpu = str2num (arg);
    otherwise
      fprintf ('[W] Unkown option %s [%s]\n', opt, arg);
  end
  
end

end