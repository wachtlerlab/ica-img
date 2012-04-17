function [ epsilon ] = gradientGetEpsilon (gradient, i)
%GRADIENTGETEPSILON Derive current eps from the gradient

epsilon = interpIter (i, gradient.iterPoints, gradient.epsilon);

end

