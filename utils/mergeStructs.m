function [ base ] = mergeStructs(base, override)

keys = fieldnames (override);

for n = 1:length(keys)
  key = char (keys(n));
  
  if isfield (base, key) && isempty (override.(key))
      base = rmfield(base, key);
  elseif isfield (base, key) && isstruct (override.(key))
      base.(key) = mergeStructs(base.(key), override.(key));  
  else
      base.(key) = override.(key);
  end
   
end

end

