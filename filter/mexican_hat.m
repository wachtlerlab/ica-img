function [ mhat ] = mexican_hat(m_size, ei_ratio, Se, Si)
%MEXICAN_HAT Summary of this function goes here
%   Detailed explanation goes here

S = 100;

[X, Y] = meshgrid ((1:m_size) - round(m_size/2));
[THETA, R] = cart2pol (X, Y);

e_gauss = do_gauss(Se, R);
i_gauss = do_gauss(Si, R);

mhat = S*(e_gauss - ei_ratio * i_gauss);

end


%% gauss

function [out] = do_gauss (S, R)

out = 1/(2*pi*S^2)*exp(-R.^2/(2*S^2));

end
