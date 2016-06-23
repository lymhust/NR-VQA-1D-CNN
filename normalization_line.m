function nsig = normalization_line(sig)

minval = min(sig);
maxval = max(sig) - minval;
tmp = bsxfun(@minus, sig, minval);
nsig = bsxfun(@rdivide, tmp, maxval);

% c = 0.01;
% miu = mean(sig);
% tmp = bsxfun(@minus, sig, miu);
% sigma = sqrt(sum(tmp.^2)./size(sig,1));
% nsig = bsxfun(@rdivide, tmp, sigma+c);

end
