function [D] = toSMLJ(mx);
% toSML -- 
%   Usage
%     [x] = toSML(mx)
%   Inputs
%          mx : 31xN matrix
%   Outputs
%             x : 4xN matrix


load SMHIJL.dat

%              S M L J
SMILmx=SMHIJL([1 2 6 5],:);
D = SMILmx*mx;


