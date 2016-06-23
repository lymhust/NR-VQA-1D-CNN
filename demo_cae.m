clear; close all; clc;
load PPG_BP_data_039;
gnum = 3;
pnum = size(PPGALL,2)/gnum;
PPG = cell(gnum, 1);
ABP = cell(gnum, 1);
for i = 1 : gnum
    PPG{i}{1} = PPGALL(:,((i - 1) * pnum + 1) : (i) * pnum);
    ABP{i}{1} = BPALL(:,((i - 1) * pnum + 1) : (i) * pnum);
end

%CAE structure
scae = {
    struct('outputmaps', 20, 'inputkernel', 63, 'outputkernel', 63, 'scale', 2, 'sigma', 1e-7, 'momentum', 0.9, 'noise', 0)
};

opts.rounds     = 500;
opts.batchsize  =    50;
opts.alpha      = 1e-2;
opts.ddinterval =   100;
opts.ddhist     =  0.5;
scae = scaesetup(scae, PPG, opts);
scae = scaetrain(scae, PPG, ABP, opts);
cae = scae{1};

%Visualize the average reconstruction error
plot(cae.rL);

%Visualize the output kernels
ff=[];
add = 0;
for i=1:numel(cae.ok{1}); 
    mm = cae.ok{1}{i}; 
    ff = [ff mm];
end; 
figure;imagesc(ff);

%Visualize output
num = 1;
x1{1} = PPG{1}{1}(:, num);
x2{1} = ABP{1}{1}(:, num);
cae = caeup(cae, x1);
cae = caedown(cae);
output = [x1{1} x2{1} cae.o{1}(:,1)];
plot(output);legend('PPG','ABP','PRE');