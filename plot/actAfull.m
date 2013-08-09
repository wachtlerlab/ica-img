function [ act ] = actAfull(model, aweight)

if ~exist('aweight', 'var') || isempty(aweight)
   aweight = 0.03; 
end

act = calcImgActivations(model);
act = actFilterWeights(act, aweight);
act = calcActATA(act);

end

