function [ Act ] = actFilterWeights(Act, border)

if ~exist('border', 'var'); border = 0.1; end;

nbf = size(Act.w, 2);
nimg = size(Act.offset, 1);
offset = Act.offset;

perimg = @(data, idx, func) cell2mat(arrayfun(...
    @(cimg) func(data(idx(cimg, 1):idx(cimg, 2), :))', 1:length(idx), ...
    'UniformOutput', false));

wfilter = zeros(size(Act.w));

epp = zeros([nbf, nimg, 3]);
wall = Act.w;

tic;
for n=1:nbf
    
    fprintf(' %03d', n);
    
    w = wall(:, n);
    
    wfbf = zeros(size(w, 1)/nimg, nimg);
    
    for img=1:nimg
        %irange = offset(img, 1):offset(img, 2);
        wi = w(offset(img, 1):offset(img, 2));
        
        %[epp_, wf, bounds] = filterExpwerFit(wi, 0.1);
        [epp_, wf] = filterExpwerFit(wi, border);
        epp(n, img, :) = epp_;
        wfbf(:, img) = wf;
        %wfilter(irange, n) = wf;
    end
    
    wfilter(:, n) = wfbf(:);
    
%      if any(wfilter(:, n) - wfbf(:))
%         warning('nonmatch'); 
%      end  
%      fprintf('\n\n')
%     
    fprintf('\b\b\b\b');
end

Act.epp = epp;
Act.wfilter = logical(wfilter);

tcalc = toc;
fprintf('t: %f\n', tcalc);

end

        %wc = wi - mean(wi);
        %stdse =  median(abs(wc))/0.6745;
        %wfilter(irange, n) = abs(wc) > 1.5*stdse;

    %wc = perimg(w, offset, @(x) (x - mean(x))');
    %stdse = perimg(wc, offset, @(x) median(abs(x))/0.6745);
    %idx = offset;
    %wfilter(:, n) = cell2mat(arrayfun(@(cimg) (abs(wc(idx(cimg, 1):idx(cimg, 2))) > 2*stdse(cimg))', 1:length(idx),'UniformOutput', false))';
    