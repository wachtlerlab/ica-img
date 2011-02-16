function [D] = toSMLI(mx);
% toSML -- 
%   Usage
%     [x] = toSML(mx)
%   Inputs
%          mx : 31xN matrix
%   Outputs
%             x : 4xN matrix


load SMHIJL.dat

%              S M L I
SMILmx=SMHIJL([1 2 6 4],:);
D = SMILmx*mx;


