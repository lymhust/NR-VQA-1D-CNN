function scae = scaesetup(cae, x, opts)
x = x{1};
cae = cae{1};
bounds = cae.outputmaps * cae.inputkernel + numel(x) * cae.outputkernel;
for j = 1 : cae.outputmaps   %  activation maps
    cae.a{j} = zeros(size(x{1},1) + cae.inputkernel - 1, 1);
    for i = 1 : numel(x)    %  input map
        cae.ik{i}{j}  = (rand(cae.inputkernel, 1)  - 0.5) * 2 * sqrt(6 / bounds);
        cae.ok{i}{j}  = (rand(cae.outputkernel, 1) - 0.5) * 2 * sqrt(6 / bounds);
        cae.vik{i}{j} = zeros(size(cae.ik{i}{j}));
        cae.vok{i}{j} = zeros(size(cae.ok{i}{j}));
    end
    cae.b{j} = 0;
    cae.vb{j} = zeros(size(cae.b{j}));
end

cae.alpha = opts.alpha;

cae.i = cell(numel(x), 1);
cae.o = cae.i;

for i = 1 : numel(cae.o)
    cae.c{i}  = 0;
    cae.vc{i} = zeros(size(cae.c{i}));
end

ss = cae.outputkernel;

cae.edgemask = ones(size(x{1}, 1), opts.batchsize);

%cae.edgemask(ss : end - ss + 1,:) = 1;

scae{1} = cae;

end

   
