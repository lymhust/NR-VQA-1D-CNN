function [cost,grad] = CNNCost(saeTheta, batch_x, batch_y, nn)

nn = vector2cnnpara(nn, saeTheta);
nn = cnnff(nn, batch_x);
nn = cnnbp(nn, batch_y);
cost = nn.L;
grad = cnndiff2vector(nn);

end