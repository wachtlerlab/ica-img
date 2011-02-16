function [D] = toSMLL(mx);
% toSML -- 
%   Usage
%     [x] = toSML(mx)
%   Inputs
%          mx : 31xN matrix
%   Outputs
%             x : 4xN matrix


load SMHIJL.dat

%              S M L L
SMILmx=SMHIJL([1 2 6 6],:);
D = SMILmx*mx;


