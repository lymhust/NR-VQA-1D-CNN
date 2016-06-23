function x = cnnffupdate(net, x, L)
net.layers{1}.a{1} = x;
inputmaps = 1;

for l = 2 : L+1   %  for each layer
    if strcmp(net.layers{l}.type, 'c')
        %  !!below can probably be handled by insane matrix operations
        for j = 1 : net.layers{l}.outputmaps   %  for each output map
            %  create temp output map
            z = zeros(size(net.layers{l - 1}.a{1}) - [net.layers{l}.kernelsize - 1 0]);
            for i = 1 : inputmaps   %  for each input map
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
            z = convn(net.layers{l - 1}.a{j}, ones(net.layers{l}.scale,1) / (net.layers{l}.scale), 'valid');   %  !! replace with variable
            net.layers{l}.a{j} = z(1 : net.layers{l}.scale : end, :);
        end
    end
end

x = net.layers{l}.a;

end
