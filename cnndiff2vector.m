function para = cnndiff2vector(net)
para = [];
for l = 2 : numel(net.layers)
    if strcmp(net.layers{l}.type, 'c')
        for j = 1 : numel(net.layers{l}.a)
            for ii = 1 : numel(net.layers{l - 1}.a)
                para = [para; net.layers{l}.dk{ii}{j}(:)];
            end
            para = [para; net.layers{l}.db{j}(:)];
        end
    end
end

para = [para; net.dffW(:)]; 
para = [para; net.dffb(:)]; 
end
