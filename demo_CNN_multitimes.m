clc; clear; close all;
%% Data pre-processing
load PPG_BP_data_all_PLE;
datanum = floor(size(PPGALL,2)/1000)*1000;
datanum = 60000;%floor(datanum * 0.8);
ind = randperm(datanum);
PPGALL = PPGALL(:,ind);
BPALL = BPALL(:,ind);
BPNUMALL = BPNUMALL(:,ind);
HRNUMALL = HRNUMALL(ind);
PLENUMALL = PLENUMALL(ind);
%% Normalization
PPGALL = normalization_line(PPGALL);
BPALL = normalization_line(BPALL);
[BPnew, minbp, maxbp] = normalization(BPNUMALL);
[HRnew, minhr, maxhr] = normalization(HRNUMALL);
PLEnew = normalization(PLENUMALL);
train_x = PPGALL;
train_y = [BPnew;HRnew;PLEnew];
train_abp = PPGALL;

%% Train CNN
% outputnum = [2 3 4 5 6 7 8 9 10 11];
outputnum = 5;
testtime = 5;
para = [64 4 32 4];
[para, outputsize] = checkpara(para);
MSESBPALL = [];
MSEDBPALL = [];

for j = 1:length(outputnum)
    MSESBP = [];
    MSEDBP = [];
    
    cnn.layers = {
        struct('type', 'i') %input layer
        struct('type', 'c', 'outputmaps', outputnum(j), 'kernelsize',para(1)) %convolution layer
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
    cnn.output = 'sigm'; %  output unit 'sigm' (=logistic), 'softmax' and 'linear'
    opts.batchsize = 1000;
    opts.numepochs = 50;
    
    cnn = cnnsetup(cnn, train_x, train_y, train_abp);
    [cnn, L] = cnntrain_modify(cnn, train_x, train_y, opts);
    plot(L);
    
    for i = 1:testtime
        %% Testing
        ind1 = randperm(datanum);
        tnum = ceil(datanum * 0.2);
        ind1 = ind1(1:tnum);
        t_x = train_x(:,ind1);
        t_y = train_y(:,ind1);
        pred = cnntest(cnn, t_x, t_y);
        plotresults(t_y, pred, maxbp, minbp, maxhr, minhr);
        VisulazeKernels(cnn);
        t_y = t_y * maxbp + minbp;
        pred = pred * maxbp + minbp;
        MSE1 = mean((t_y(1,:) - pred(1,:)).^2);
        MSE2 = mean((t_y(2,:) - pred(2,:)).^2);
        MSESBP = [MSESBP;MSE1];
        MSEDBP = [MSEDBP;MSE2];
    end
    MSESBPALL = [MSESBPALL MSESBP];
    MSEDBPALL = [MSEDBPALL MSEDBP];
    disp(j);disp(i);
end

%% Plotting
load MSErandom10;
SBPrand = sort(MSESBPALL);DBPrand = sort(MSEDBPALL);
SBPrand = SBPrand(5,2:10);DBPrand = DBPrand(5,2:10);
load MSEpretrain10;
SBPpre = sort(MSESBPALL);DBPpre = sort(MSEDBPALL);
SBPpre = SBPpre(5,2:10);DBPpre = DBPpre(5,2:10);
figure;plot([SBPrand' DBPrand' SBPpre' DBPpre']);legend('SBP Random','DBP Random','SBP Pre-train','DBP Pre-train');



