function net = cnnbp(net, y)
n = numel(net.layers);

%   error
net.e = net.o - y;
    
switch net.output
    case 'sigm'
        net.L = 1/2 * sum(net.e(:) .^ 2) / size(net.e, 2);
        net.od = net.e .* (net.o .* (1 - net.o));
    case 'linear'
        net.L = 1/2 * sum(net.e(:) .^ 2) / size(net.e, 2);
        net.od = net.e;
    case 'softmax'
        net.L = -sum(sum(y .* log(net.o))) / size(net.e, 2);
        net.od = net.e;
end

%%  backprop deltas
net.fvd = (net.ffW' * net.od);              %  feature vector delta
if strcmp(net.layers{n}.type, 'c')         %  only conv layers has sigm function
    net.fvd = net.fvd .* (net.fv .* (1 - net.fv));
end

%  reshape feature vector deltas into output map style
sa = size(net.layers{n}.a{1});
fvnum = sa(1);
for j = 1 : numel(net.layers{n}.a)
    net.layers{n}.d{j} = net.fvd(((j - 1) * fvnum + 1) : j * fvnum, :);
end

for l = (n - 1) : -1 : 1
    if strcmp(net.layers{l}.type, 'c')
        for j = 1 : numel(net.layers{l}.a)
            % For mean pooling
            %               net.layers{l}.d{j} = net.layers{l}.a{j} .* (1 - net.layers{l}.a{j}) .* (expand(net.layers{l + 1}.d{j}, [net.layers{l + 1}.scale 1]) / net.layers{l + 1}.scale);
            % For max pooling
            pos = net.layers{l+1}.pos{j};
            tmppos = [pos(:); net.layers{l + 1}.scale];
            mask = full(sparse(tmppos, 1:length(tmppos), 1));
            mask = mask(:,1:end-1);
            mask = reshape(mask(:),[size(pos,1)*net.layers{l + 1}.scale size(pos,2)]);           
            tmpd = expand(net.layers{l + 1}.d{j}, [net.layers{l + 1}.scale 1]) .* mask;
            net.layers{l}.d{j} = net.layers{l}.a{j} .* (1 - net.layers{l}.a{j}) .* (tmpd);
        end
    elseif strcmp(net.layers{l}.type, 's')
        for i = 1 : numel(net.layers{l}.a)
            z = zeros(size(net.layers{l}.a{1}));
            for j = 1 : numel(net.layers{l + 1}.a)
                z = z + convn(net.layers{l + 1}.d{j}, rot180(net.layers{l + 1}.k{i}{j}), 'full');
            end
            net.layers{l}.d{i} = z;
        end
    end
end

%%  calc gradients
for l = 2 : n
    if strcmp(net.layers{l}.type, 'c')
        for j = 1 : numel(net.layers{l}.a)
            for i = 1 : numel(net.layers{l - 1}.a)
                net.layers{l}.dk{i}{j} = convn(flipall(net.layers{l - 1}.a{i}), net.layers{l}.d{j}, 'valid') / size(net.layers{l}.d{j}, 2);
            end
            net.layers{l}.db{j} = sum(net.layers{l}.d{j}(:)) / size(net.layers{l}.d{j}, 2);
        end
    end
end
net.dffW = net.od * (net.fv)' / size(net.od, 2);
net.dffb = mean(net.od, 2);

    function X = rot180(X)
        X = flipdim(flipdim(X, 1), 2);
    end
end
