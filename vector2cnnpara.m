function net = vector2cnnpara(net, para)
startind = 1;
for l = 2 : numel(net.layers)
    if strcmp(net.layers{l}.type, 'c')
        for j = 1 : numel(net.layers{l}.a)
            for ii = 1 : numel(net.layers{l - 1}.a)
                len = numel(net.layers{l}.k{ii}{j});
                tmppara = para(startind:startind+len-1);
                net.layers{l}.k{ii}{j} = reshape(tmppara,size(net.layers{l}.k{ii}{j}));
                startind = startind +len;
            end           
            len = numel(net.layers{l}.b{j});
            tmppara = para(startind:startind+len-1);
            net.layers{l}.b{j} = reshape(tmppara,size(net.layers{l}.b{j}));
            startind = startind +len;
        end
    end
end
len = numel(net.ffW);
tmppara = para(startind:startind+len-1);
net.ffW = reshape(tmppara,size(net.ffW));
startind = startind +len;

len = numel(net.ffb);
tmppara = para(startind:startind+len-1);
net.ffb = reshape(tmppara,size(net.ffb));
end
