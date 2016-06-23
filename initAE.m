function [ffW, ffb] = initAE(input, output)
m = size(input, 2);
batchsize = 500;
%numbatches = m / batchsize;
%assert(rem(numbatches, 1) == 0, 'numbatches must be a integer');
kk = randperm(m);

options.Method = 'lbfgs';
options.maxIter = 10;
options.display = 'on';
options.Corr = 10;
sparsityParam = 0.1;   % desired average activation of the hidden units
lambda = 3e-7;         % weight decay parameter
beta = 5; 
L = [];
inputsize = size(input,1);
outputsize = 2;
saeTheta = initializeParameters(outputsize, inputsize);

for l = 1 : 20
    batch_x = input(:,kk((l - 1) * batchsize + 1 : l * batchsize));   
    batch_y = output(:,kk((l - 1) * batchsize + 1 : l * batchsize));
    
   [saeOptThetatmp, cost, exitflag, op] = minFunc( @(p) sparseAutoencoderLinearCost_inandout(p, ...
    inputsize, outputsize, ...
    lambda, sparsityParam, ...
    beta, batch_x, batch_y), ...
    saeTheta, options);
    saeTheta = saeOptThetatmp;
    L = [L;op.trace.fval];
end

ffW = reshape(saeOptThetatmp(1:outputsize*inputsize),outputsize, inputsize);
ffb = saeOptThetatmp(2*outputsize*inputsize+1:2*outputsize*inputsize+outputsize);










