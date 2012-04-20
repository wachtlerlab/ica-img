function [ gradient ] = createGradient (cfg, maxIter)

gradient.iterPoints = cfg.iter_points;
gradient.epsilon = cfg.eps_scale * cfg.epsilon;

maxGradient = max(gradient.iterPoints);
if maxIter > maxGradient
  warning ('ICA:gradient_exceeded', ...
    'Iterations exceed gradient definition (%d < %d)', ...
    maxGradient, maxIter )
end


end

