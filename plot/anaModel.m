function [imgn] = anaModel(Model, idx)

Model = sortModelA(Model);
A = Model.A;
bf = reshape (A(:,idx), 6, 7*7);

X = [ 0,  0,  0,  0,  1, -1;    % L
      0,  0,  1, -1,  0,  0;    % M 
      1,  -1, 0,  0,  0,  0];   % S
    
imgx = permute (reshape (X*bf, 3, 7,7), [3 2 1]);
imgn = 0.5+0.5*imgx/max(abs(imgx(:)));

end
