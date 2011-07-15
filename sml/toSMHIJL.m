function [D] = toSMHIJL(mx);
% toSML -- 
%   Usage
%     [x] = toSMHIJL(mx)
%   Inputs
%          mx : 31xN matrix
%   Outputs
%             x : 6xN matrix

load SMHIJL.dat

%              S M H I J L
SMILmx=SMHIJL([1 2 3 4 5 6],:);
D = SMILmx*mx;


