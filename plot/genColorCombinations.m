function [ combinations ] = genColorCombinations (resolution)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

lspec = 1:-1*resolution:0;
[x,y,z] = meshgrid (lspec, lspec, lspec);
combinations = [y(:),x(:),z(:)];

end

