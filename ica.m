function [Model, cfg, dataset] = ica (cfg, varargin)

if nargin < 1
  error('Need configuration identifer')
end;

options = struct('nofit', 0, ...
                 'autosave', 1, ...
                 'savestate', 1, ...
                 'savefreq', 100, ...
                 'progress', 0, ...
                 'plotfreq', 50);

if nargin > 1
  options = parse_varargs (options, varargin);
end


currev = getCurRev ();

%%
fprintf ('Configuration: \n');
disp(options);

%% laod config and prepare dataset, prior, model

if (ischar (cfg))
  cfg = loadConfig (cfg);
end

fprintf ('Starting simulation for %s [code: %s]\n', cfg.id, currev);

imageset = createImageSet (cfg.data);

tstart = tic;
fprintf('\nGenerating dataset...\n');
dataset = createDataSet (imageset, cfg);
fprintf('   done in %f\n', toc(tstart));

gradient = createGradient (cfg.gradient, dataset.maxiter);

Model = setupModel(cfg, dataset);


%% check if we are just creating the PIC
if options.nofit
  datapath = saveSCAI (cfg, dataset);
  fprintf ('Created SCAI (%s)\n', datapath);
  return
end


%% create state dir if necessary
stateDir = fullfile ('..', 'state');

if exist (stateDir, 'dir') == 0
   mkdir (stateDir); 
end


%% create figures for display, if needed
if options.progress
  figure(1)
  figure(2)
  figure(3)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Infer the Model
fprintf ('\nFitting %s for config %s [%s]\n',...
  Model.id(1:7), Model.cfg(1:7), datestr (clock (), 'yyyymmddHHMM'));
tStart = tic;

[Model, Result] = fitModel (Model, gradient, dataset, options);

tDuration = toc (tStart);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 

% time reporting
Result.tDuration = tDuration;

Model.onGPU = 0;
Model.fit_time = tDuration;

fprintf (['Total time: (',num2str(Result.tDuration),')\n']);
%Model.Result = Result;
%Model.codeVersion = currev;

if options.autosave
  filename = saveSCAI (cfg, dataset, Model);
  fprintf ('Saved results to %s\n', filename);
end

end


function [options] = parse_varargs(options, varargin)

args = size (varargin{1});
for cur = 1:2:args(2)
  opt = char (varargin{1}(cur));
  arg = char (varargin{1}(cur + 1));
  
  switch opt
    case 'nofit'
      options.nofit = str2num (arg);
    case 'autosave'
      options.autosave = str2num (arg);
    case 'savestate'
      options.savestate = str2num (arg);
    case 'savefreq'
      options.savefreq = str2num (arg);
    case 'progress'
      options.progress = str2num (arg);
    case 'plotfreq'
      options.plotfreq = str2num (arg);
    otherwise
      fprintf ('[W] Unkown option %s [%s]\n', opt, arg);
  end
  
end

end