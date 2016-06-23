function [newpara,outputsize] = checkpara(para)
newpara = para;
inputsize = 39;
outputsize = zeros(1,length(para)/2);
j = 1;
for i = 1 : 2: length(para)
    tmp = (inputsize - para(i) + 1) / para(i+1);
    tmp = round(tmp);
    if(tmp < 0) error('Kernel size is too large!');end
    newpara(i) = inputsize + 1 - tmp * para(i+1);
    inputsize = tmp;
    outputsize(j) = tmp; j = j + 1;
end