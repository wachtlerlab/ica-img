function [D] = toSMLH(mx);
% toSML -- 
%   Usage
%     [x] = toSML(mx)
%   Inputs
%          mx : 31xN matrix
%   Outputs
%             x : 4xN matrix

load SMHIJL.dat

%              S M L H
SMILmx=SMHIJL([1 2 6 3],:);
D = SMILmx*mx;


