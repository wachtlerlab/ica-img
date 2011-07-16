function [Model] = startICA (modelId, varargin)

if nargin < 1
  modelId = 'color_cs_rect_1';
end;

options = struct('autosave', 'false', ...
                 'savestate', 'false', ...
                 'progress', 'true');
               
if nargin > 1
  options = parse_varargs (options, varargin);
end

currev = getCurRev ();

fprintf ('Starting simulation for %s [code: %s]', modelId, currev);
[Model, Result] = fitModel (modelId, options);

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
    otherwise
      fprintf ('[W] Unkown option %s [%s]\n', opt, arg);
  end
  
end

end