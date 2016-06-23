function cae = caeup(cae, x)
    cae.i = x;

    %init temp vars for parrallel processing
    pa  = cell(size(cae.a));
    pi  = cae.i;
    pik = cae.ik;
    pb  = cae.b;

    for j = 1 : numel(cae.a)
        z = 0;
        for i = 1 : numel(pi)
            z = z + convn(pi{i}, pik{i}{j}, 'full');
        end
        pa{j} = sigm(z + pb{j});

        %  Max pool.
        if ~isequal(cae.scale, 1)
            pa{j} = pooling(pa{j}, cae.scale);
        end

    end
    cae.a = pa;

 end
