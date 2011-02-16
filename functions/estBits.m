function b = estBits( logL, precision, D )
% estBits -- convert log likelihood into estimated bits
%   Usage
%     b = estBits( logL, precision, D )
%   Inputs
%     logL        log likelihood of pattern set
%     precision   encoding precision (in bits)
%     D           pattern set
%   Outputs
%     b           estimated bits to encode pattern set

% Written by Mike Lewicki 4/99
%
% Copyright (c) 1999 Michael S. Lewicki and CMU
%
% Permission to use, copy, modify, and distribute this software and its
% documentation for any purpose and without fee is hereby granted,
% provided that the above copyright notice and this paragraph appear in
% all copies.  Copyright holder(s) make no representation about the
% suitability of this software for any purpose. It is provided "as is"
% without express or implied warranty.

[L,N] = size(D);

dataRange = range(D(:));
delta = dataRange / (2^precision);

b = -logL - (L*N)*log2(delta);
