function [out] = sortModelA (Model)

if isfield(Model, 'A_orig')
  out = Model;
  return;
end

A_sorted = sortAbf (Model.A);
Model.A_orig = Model.A;
Model.A = A_sorted;
out = Model;

end