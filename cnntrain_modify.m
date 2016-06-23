function [nn, L] = cnntrain_modify(nn, train_x, train_y, opts)

m = size(train_x, 2);
batchsize = opts.batchsize;
numepochs = opts.numepochs;
numbatches = m / batchsize;
assert(rem(numbatches, 1) == 0, 'numbatches must be a integer');
kk = randperm(m);
%  Use minFunc to minimize the function
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
% function. Generally, for minFunc to work, you
% need a function pointer with two outputs: the
% function value and the gradient. In our problem,
% sparseAutoencoderCost.m satisfies this.
options.maxIter = numepochs;	  % Maximum number of iterations of L-BFGS to run
options.display = 'on';
options.Corr = 10;
L = [];

for l = 1 : numbatches
    batch_x = train_x(:,kk((l - 1) * batchsize + 1 : l * batchsize));
    
    %Add noise to input (for use in denoising autoencoder)
%     if(nn.inputZeroMaskedFraction ~= 0)
%         batch_x = batch_x.*(rand(size(batch_x))>nn.inputZeroMaskedFraction);
%     end
    
    batch_y = train_y(:,kk((l - 1) * batchsize + 1 : l * batchsize));
    
    nn = cnnff(nn, batch_x);
    saeTheta = cnnpara2vector(nn);
    [saeOptThetatmp, cost, exitflag, output] = minFunc( @(p) CNNCost(p, batch_x, batch_y, nn), saeTheta, options);
    nn = vector2cnnpara(nn, saeOptThetatmp);  
    L = [L;output.trace.fval];
end

end




