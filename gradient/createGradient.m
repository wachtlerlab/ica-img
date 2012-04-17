function [ gradient ] = createGradient (fitParam, maxIter)
%CREATEGRADIENT Create the gradient for updating A
%               basically a dummy function for the old config system

gradient.iterPoints = fitParam.iterPts;
gradient.epsilon = fitParam.epsilon;

maxGradient = max(gradient.iterPoints);
if maxIter > maxGradient
  warning ('ICA:gradient_exceeded', ...
    'Iterations exceed gradient definition (%d < %d)', ...
    maxGradient, maxIter )
end

end
