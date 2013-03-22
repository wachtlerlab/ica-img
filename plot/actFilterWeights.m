function [ Act ] = actFilterWeights(Act)

nbf = size(Act.w, 2);
offset = Act.offset;

perimg = @(data, idx, func) cell2mat(arrayfun(...
    @(cimg) func(data(idx(cimg, 1):idx(cimg, 2), :))', 1:length(idx), ...
    'UniformOutput', false));

wfilter = zeros(size(Act.w));

for n=1:nbf
    
    fprintf(' %03d', n);
    
    w = Act.w(:, n);
    
    for img=1:8
        irange = offset(img, 1):offset(img, 2);
        wi = w(offset(img, 1):offset(img, 2));
        wc = wi - mean(wi);
        stdse =  median(abs(wc))/0.6745;
        wfilter(irange, n) = abs(wc) > 1.5*stdse;
    end
    
    fprintf('\b\b\b\b');
end

Act.wfilter = logical(wfilter);

end



    %wc = perimg(w, offset, @(x) (x - mean(x))');
    %stdse = perimg(wc, offset, @(x) median(abs(x))/0.6745);
    %idx = offset;
    %wfilter(:, n) = cell2mat(arrayfun(@(cimg) (abs(wc(idx(cimg, 1):idx(cimg, 2))) > 2*stdse(cimg))', 1:length(idx),'UniformOutput', false))';
    