clc; clear; close all;
%% Data pre-processing
% Train data
load FaceFeaturesGray3D;
% LABELS(find(LABELS>1))=2;
classnum = length(unique(LABELS));
totalnum = length(LABELS);
train_x = normalization_line(IMGFEATURE);
train_y = full(sparse(LABELS, 1:totalnum, 1));
nn = randperm(totalnum);
train_x = train_x(:,nn);
train_y = train_y(:,nn);
LABELS = LABELS(nn);
% Test data
test_x = train_x;
test_y = train_y;

%% ex1 Train a 6c-2s-12c-2s Convolutional neural network
para = [8 2 4 2];
[para, outputsize] = checkpara(para);

cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 5, 'kernelsize',para(1)) %convolution layer
    struct('type', 's', 'scale', para(2)) %subsampling layer
    struct('type', 'c', 'outputmaps', 10, 'kernelsize', para(3)) %convolution layer
    struct('type', 's', 'scale', para(4)) %subsampling layer
%     struct('type', 'c', 'outputmaps', 24, 'kernelsize', para(5)) %convolution layer
%     struct('type', 's', 'scale', para(6)) %subsampling layer
    
    };
% layer有三种，i是input，c是convolution，s是subsampling
% 'c'的outputmaps是convolution之后有多少张图，比如上(最上那张经典的))第一层convolution之后就有六个特征图
% 'c'的kernelsize 其实就是用来convolution的patch是多大
% 's'的scale就是pooling的size为scale*scale的区域
cnn.output = 'softmax'; %  output unit 'sigm' (=logistic), 'softmax' and 'linear'

opts.alpha = 1e-2;
opts.batchsize = 56;
opts.numepochs = 800;

cnn = cnnsetup(cnn, train_x, train_y, train_x);
[cnn, L] = cnntrain_modify(cnn, train_x, train_y, opts);
plot(L);
% Testing
pred = cnntest(cnn, test_x, test_y);
[val,predclass] = max(pred);
acc = mean(LABELS == predclass')
cfmat = cfmatrix(LABELS,predclass');
cfmat = bsxfun(@rdivide, cfmat, sum(cfmat));
names = ['Real ';'Photo';'Blink';'Video'];
draw_cm(cfmat,names,4);        
       
% Visulazation
VisulazeKernels(cnn);


