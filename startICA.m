function [Model] = startICA (modelId, varargin)

if nargin < 1
  modelId = 'color_cs_rect_1';
end;

options = struct('createpic', 0, ...
                 'autosave', 1, ...
                 'savestate', 1, ...
                 'progress', 0);
ds_path = '';

if nargin > 1
  [options, ds_path] = parse_varargs (options, varargin);
end


if exist (fullfile ('config', [modelId '.m']), 'file') == 0
  error ('Cannot find config');
end

currev = getCurRev ();

fprintf ('Starting simulation for %s [code: %s]', modelId, currev);

%%
% basic init
clear Model fitPar dispPar Result;


[Model, fitPar, dispPar, dataPar] = loadConfig (modelId);

Model.id = DataHash (Model, struct ('Method', 'SHA-1'));


%% Prepare image data

if isempty (ds_path)

  images = prepareImages (dataPar);

  tstart = tic;
  fprintf('\nGenerating dataset...\n');
  dataset = createDataSet (images, fitPar, dataPar);
  fprintf('   done in %f\n', toc(tstart));
  
else
  dataset = loadDataSet (ds_path, '/ds');
  Model.A = dataset.Ainit;
end

%% check if we are just creating the PIC
if options.createpic
  datapath = savePIC (modelId, Model, dataset, fitPar);
  fprintf ('Created PIC (%s)\n', datapath);
  return
end


%% create state dir if necessary
stateDir = fullfile ('..', 'state');

if exist (stateDir, 'dir') == 0
   mkdir (stateDir); 
end


%% create figures for display, if needed
dispPar.plotflag = options.progress;
fitPar.saveflag = options.savestate;

if options.progress
  figure(1)
  figure(2)
  figure(3)
end

%% Infer the Model
fprintf ('\nFitting %s for config %s [%s]\n',...
  Model.id(1:7), Model.cfgId(1:7), datestr (clock (), 'yyyymmddHHMM'));

tStart = tic;

[Model, Result] = fitModel (Model, fitPar, dispPar, dataset, options);

tDuration = toc (tStart);
%% 

% time reporting
Result.tDuration = tDuration;

Model.fitPar = fitPar;
Model.dispPar = dispPar;
Model.dataPar = dataPar;
Model.onGPU = 0;
Model.dataset = dataset;

fprintf (['Total time: (',num2str(Result.tDuration),')\n']);
Model.Result = Result;
Model.codeVersion = currev;

if options.autosave
  filename = saveResult (Model);
  fprintf ('Saved results to %s\n', filename);
end

end


function [options, dataset] = parse_varargs(options, varargin)

args = size (varargin{1});
dataset = '';
for cur = 1:2:args(2)
  opt = char (varargin{1}(cur));
  arg = char (varargin{1}(cur + 1));
  
  switch opt
    case 'createpic'
      options.createpic = str2num (arg);
    case 'autosave'
      options.autosave = str2num (arg);
    case 'savestate'
      options.savestate = str2num (arg);
    case 'progress'
      options.progress = str2num (arg);
    case 'dataset'
      dataset = arg;
    otherwise
      fprintf ('[W] Unkown option %s [%s]\n', opt, arg);
  end
  
end

end