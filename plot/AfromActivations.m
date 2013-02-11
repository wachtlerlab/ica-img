function [A] = AfromActivations(Model, Act, imgnr)

ps    = 7;
nchan = 3;

A = struct();
A.name    = [Model.cfg(1:7) '-' Model.id(1:7)];
A.id      = Model.id;
A.nchan   = nchan;

nbf = length(Act{1}.bf);

Ainit = zeros(ps*ps*nchan, nbf);
A.sml = Ainit;
A.lms = Ainit;
A.rgb = Ainit;

A.dkl = zeros(ps*ps, 3, nbf);
A.pcs = zeros(2, nbf);

bfact = zeros(ps*ps*nchan, nbf, length(imgnr));

for imn=1:length(imgnr)
    ix = imgnr(imn);
    asm = cell2mat(Act{ix}.bf);
    for ibf=1:nbf
        omraw = asm(ibf).omraw;
        %omraw = omraw - mean(omraw(:));
        %omraw = omraw / sqrt(var(omraw(:)));
        bfact(:, ibf, imn) = omraw;
    end
end

meanact = mean(bfact, 3);

for n=1:nbf
    ompatch = meanact(:, n);
    sml = reshape(ompatch, nchan, ps, ps);
    lms = flipdim(reshape(ompatch, nchan, ps, ps), 1);
    rgb = 0.5 + 0.5 * (lms/max(abs(lms(:))));
    
    slice = permute(reshape(rgb, nchan, ps*ps), [2 1]);
    L =  slice(:, 1);
    M =  slice(:, 2);
    S =  slice(:, 3);

    x = L-M;
    y = S-((L+M)/2);
    z = L+M;
    
    A.dkl(:, :, n) = [x, y, z];
    
    [pc,~,latent,~] = princomp([x y]);
    pc = pc(:,1) * latent (1);
    A.pcs(:, n) = pc;
    
    A.sml(:, n) = sml(:);
    A.lms(:, n) = lms(:);
    A.rgb(:, n) = rgb(:);
end


end

%  ompatch = mean(opatchw, 2);
%     sml = reshape(flipdim(reshape(ompatch, 3, ps, ps), 1), bflen, 1);
%     rgb = 0.5 + 0.5 * (sml/max(abs(sml)));
% 
