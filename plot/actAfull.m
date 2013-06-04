function [ act ] = actAfull(model)

act = calcImgActivations(model);
act = actFilterWeights(act, 0.03);
act = calcActATA(act);

end

