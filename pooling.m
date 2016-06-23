function poolsig = pooling(sig, scale)
[len, datanum] = size(sig);
num = ceil(len/scale);
if(mod(sig,scale))
    addnum = num * scale - len;
    ind = len-1:-1:len-addnum;
    sigadd = sig(ind,:);
    sig = [sig;sigadd];
end
poolsig = zeros(num*scale,datanum);
for i = 1:scale:num*scale-scale+1
    sigtmp = sig(i:i+scale-1,:);
    [val, ind] = max(sigtmp);
    ind = [ind scale];
    mask = full(sparse(ind, 1:datanum+1, 1));
    sigtmp = sigtmp .* mask(:,1:datanum);
    poolsig(i:i+scale-1,:) = sigtmp;
end
poolsig = poolsig(1:len,:);
