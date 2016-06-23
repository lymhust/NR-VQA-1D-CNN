function net = cnnsetup(net, x, y, abp)
assert(~isOctave() || compare_versions(OCTAVE_VERSION, '3.8.0', '>='), ['Octave 3.8.0 or greater is required for CNNs as there is a bug in convolution in previous versions. See http://savannah.gnu.org/bugs/?39314. Your version is ' myOctaveVersion]);
inputmaps = 1;
mapsize = size(x(:,1),1);

for l = 1 : numel(net.layers)   %  layer
    if strcmp(net.layers{l}.type, 's')
        mapsize = mapsize / net.layers{l}.scale;
        assert(all(floor(mapsize)==mapsize), ['Layer ' num2str(l) ' size must be integer. Actual: ' num2str(mapsize)]);
        for j = 1 : inputmaps
            net.layers{l}.b{j} = 0;
        end
    end
    if strcmp(net.layers{l}.type, 'c')
        mapsize = mapsize - net.layers{l}.kernelsize + 1;
        % ���ز�Ĵ�С����һ��(�������ͼ����)*(���������patchͼ�Ĵ�С)
        %% �Դ������ʼ��
        fan_out = net.layers{l}.outputmaps * net.layers{l}.kernelsize;
        for j = 1 : net.layers{l}.outputmaps  %  output map
            fan_in = inputmaps * net.layers{l}.outputmaps;
            % ����ÿһ���������ͼ���ж��ٸ���������ǰ��
            for i = 1 : inputmaps  %  input map
                % ��ʼ���������ϵ��������Ȩ��
                %                 r  = sqrt(6) / sqrt(fan_in + fan_out);   % we'll choose weights uniformly from the interval [-r, r]
                %                 net.layers{l}.k{i}{j} = rand(net.layers{l}.kernelsize,1) * 2 * r - r;
                net.layers{l}.k{i}{j} = (rand(net.layers{l}.kernelsize,1) - 0.5) * 2 * sqrt(6 / (fan_in + fan_out));
                
            end
            net.layers{l}.b{j} = 0;
        end
        %% ����CAE��ʼ��
%         for i = 1 : inputmaps
%             if inputmaps == 1
%                 [net.layers{l}.k(i),net.layers{l}.b] = initCAE(x, abp, net.layers{l}.outputmaps, net.layers{l}.kernelsize);
%                 xori = x;
%                 abpori = abp;
%             else
%                 [net.layers{l}.k(i),net.layers{l}.b] = initCAE(x{i}, abp{i}, net.layers{l}.outputmaps, net.layers{l}.kernelsize);
%             end
%         end
%         x = cnnffupdate(net, xori, l);
%         abp = cnnffupdate(net, abpori, l);
        %%  �����������ݼ�����
        inputmaps = net.layers{l}.outputmaps;
    end
end
% 'onum' is the number of labels, that's why it is calculated using size(y, 1). If you have 20 labels so the output of the network will be 20 neurons.
% 'fvnum' is the number of output neurons at the last layer, the layer just before the output layer.
% 'ffb' is the biases of the output neurons.
% 'ffW' is the weights between the last layer and the output neurons. Note that the last layer is fully connected to the output layer, that's why the size of the weights is (onum * fvnum)
%% �Դ������ʼ�����һ��
fvnum = mapsize * inputmaps;
onum = size(y, 1);
net.ffb = zeros(onum, 1);
net.ffW = (rand(onum, fvnum) - 0.5) * 2 * sqrt(6 / (onum + fvnum));
r = sqrt(6 / (onum + fvnum));
net.ffW = rand(onum, fvnum) * 2 * r - r;
%% ����AE��ʼ�����һ��
% X = [];
% Y = [];
% for j = 1 : numel(x)
%     sa = size(x{j});
%     X = [X; reshape(x{j}, sa(1), sa(2))];
%     Y = [Y; reshape(x{j}, sa(1), sa(2))];
% end
% [net.ffW, net.ffb] = initAE(X, Y);

end
