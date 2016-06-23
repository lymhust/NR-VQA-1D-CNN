function [nsig, minval, maxval] = normalization(sig, minval, maxval)
%% normalize 0 to 1
[m,n] = size(sig);
sig = sig(:);
if nargin < 3
    minval = min(sig);
    maxval = max(sig) - minval;
    tmp = sig - minval;
    tmp = tmp / maxval;
    nsig = reshape(tmp,[m,n]);
else
    minval = minval;
    maxval = maxval;
    tmp = sig - minval;
    tmp = tmp / maxval;
    nsig = reshape(tmp,[m,n]);
end

end
