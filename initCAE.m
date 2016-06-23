function [weights, bias] = initCAE(train_x, train_abp, outputmaps, kernelsize)
gnum = 10;
pnum = size(train_x,2)/gnum;
x = cell(gnum, 1);
y = cell(gnum, 1);
for i = 1 : gnum
    x{i}{1} = train_x(:,((i - 1) * pnum + 1) : (i) * pnum);
    y{i}{1} = train_abp(:,((i - 1) * pnum + 1) : (i) * pnum);
end

%CAE structure
scae = {
    struct('outputmaps', outputmaps, 'inputkernel', kernelsize, 'outputkernel', kernelsize, 'scale', 2, 'sigma', 1e-1, 'momentum', 0.9, 'noise', 0.5)
};
opts.rounds     = 500;
opts.batchsize  = 500;
opts.alpha      = 1e-2;
opts.ddinterval =   10;
opts.ddhist     =  0.5;
scae = scaesetup(scae, x, opts);
scae = scaetrain(scae, x, y, opts);
cae = scae{1};
weights = cae.ik;
bias = cae.b;

% %Visualize the average reconstruction error
% plot(cae.rL);
% 
% %Visualize the output kernels
% ff=[];
% for i=1:numel(cae.ok{1});
%     mm = cae.ok{1}{i}(1,:,:);
%     ff(i,:) = mm(:);
% end;
% figure;visualize(ff')

end
