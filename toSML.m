function [D] = toSML(mx);
% toSML -- 
%   Usage
%     [x] = toSML(mx)
%   Inputs
%          mx : 31xN matrix
%   Outputs
%             x : 3xN matrix

% 090704: renamed original toSML.m to toSML.m_010917


load SMHIJL.dat

%              S M L 
SMLmx=SMHIJL([1 2 6],:);
D = SMLmx*mx;


