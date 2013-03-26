function [ act ] = actAfull(model)

act = calcImgActivations(model);
act = actFilterWeights(act, 0.05);
act = calcActATA(act);

end

