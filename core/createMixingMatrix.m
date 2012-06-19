function [ A ] = createMixingMatrix(cfg, dim)

mcfg = cfg.init_guess;

if strcmpi(mcfg.type, 'randn')
  A = randn(dim, dim);
  
  if isfield (mcfg, 'bias')
    A = A + mcfg.bias * eye (dim, dim);
  end
  
elseif strcmpi(mcfg.type, 'identity')
    A = eye (dim, dim);
end

if isfield (mcfg, 'norm') && mcfg.norm
  for n = 1:dim
    A(:,n) = A(:,n)/abs(max(A(:,n)));
  end
end

end

