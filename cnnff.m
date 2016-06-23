function net = cnnff(net, x)
    n = numel(net.layers);
    net.layers{1}.a{1} = x;
    inputmaps = 1;

    for l = 2 : n   %  for each layer
        if strcmp(net.layers{l}.type, 'c')
            %  !!below can probably be handled by insane matrix operations
            for j = 1 : net.layers{l}.outputmaps   %  for each output map
                %  create temp output map
                z = zeros(size(net.layers{l - 1}.a{1}) - [net.layers{l}.kernelsize - 1 0]);
                for  i = 1 : inputmaps   %  for each input map
                    %  convolve with corresponding kernel and add to temp output map
                    z = z + convn(net.layers{l - 1}.a{i}, net.layers{l}.k{i}{j}, 'valid');
                end
                %  add bias, pass through nonlinearity
                net.layers{l}.a{j} = sigm(z + net.layers{l}.b{j});
            end
            %  set number of input maps to this layers number of outputmaps
            inputmaps = net.layers{l}.outputmaps;
        elseif strcmp(net.layers{l}.type, 's')
            %  downsample
            for j = 1 : inputmaps
                % mean pooling
%                   z = convn(net.layers{l - 1}.a{j}, ones(net.layers{l}.scale,1) / (net.layers{l}.scale), 'valid');   %  !! replace with variable
%                  net.layers{l}.a{j} = z(1 : net.layers{l}.scale : end, :);
                % max pooling
                z = reshape(net.layers{l - 1}.a{j},[net.layers{l}.scale size(net.layers{l - 1}.a{j},1)/net.layers{l}.scale size(net.layers{l - 1}.a{j},2)]);
                [z,pos] = max(z);
                z = squeeze(z);
                pos = squeeze(pos);
                net.layers{l}.a{j} = z;
                net.layers{l}.pos{j} = pos;
            end
        end
    end

    %  concatenate all end layer feature maps into vector
    net.fv = [];
    for j = 1 : numel(net.layers{n}.a)
        net.fv = [net.fv; net.layers{n}.a{j}];
    end
    %  feedforward into output perceptrons
    switch net.output 
        case 'sigm'
            net.o = sigm(net.ffW * net.fv + repmat(net.ffb, 1, size(net.fv, 2)));
        case 'linear'
            net.o = net.ffW * net.fv + repmat(net.ffb, 1, size(net.fv, 2));
        case 'softmax'
            net.o = net.ffW * net.fv;
            net.o = exp(net.o);
            net.o = bsxfun(@rdivide, net.o, sum(net.o)); 
    end

end
