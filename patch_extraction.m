function [imblock,blocklabels,blockscores,blocknames] = patch_extraction(samples,winsize,folder,MOS,channel,database)

Files = dir(strcat(folder,'*.*'));
dataNum = length(Files) - 2;
getsample = floor(samples/dataNum);
imblock = zeros(winsize*winsize*channel,samples);
blocklabels = zeros(samples,1);
blockscores = zeros(samples,1);
blocknames = cell(samples,1);
sampleNum = 1;

for num_file = 3:length(Files) % Traverse image folder
    
    % Even things out (take enough from last image)
    if num_file == dataNum+2, getsample = samples - sampleNum + 1; end
    % Load the image. Change the path here if needed.
    str = Files(num_file).name;
    I = imread(strcat(folder,str));
    if (channel == 1)
        I = rgb2gray(I);
    end
    
    switch database
        case 'L'
%             if (num_file == 3)
%                 Iori = I;
%             end
            mos = MOS(num_file-2);
        case 'M'
%             if (num_file == 3)
%                 Iori = I;
%             end
            ind = find(strcmp(distimgs,str));
            if isempty(ind)
                mos = 85;
            else
                mos = 100 - MOS(ind);
            end
        case 'T'
%             if (num_file == 3)
%                 oriname = str(1:3);
%                 oriadd = sprintf('.\\tid2008 - modified\\reference_images\\%s.BMP',oriname);
%                 Iori = imread(oriadd);
%                 Iori = rgb2gray(Iori);
%                 Iori = double(Iori);
%             end
            str(1) = 'I';
            ind = find(strcmp(distimgs,str));
            if isempty(ind)
                mos = -1;
            else
                mos = MOS(ind) * 10 + 15;
            end
        otherwise
            disp('ERROR');
            break;
    end
      
    % Sample patches in random locations
    sizex = size(I,2);
    sizey = size(I,1);
    posx = floor(rand(1,getsample)*(sizex-winsize-2))+1;
    posy = floor(rand(1,getsample)*(sizey-winsize-1))+1;
    for j=1:getsample
        Iblock = I(posy(1,j):posy(1,j)+winsize-1,posx(1,j):posx(1,j)+winsize-1,:);
        Iblock = imgpreprocessing(Iblock);
        imblock(:,sampleNum) = Iblock;      
      
        blockscores(sampleNum) = mos;
        blocklabels(sampleNum) = resetscores(mos);
        inds = strfind(folder,'\');
        name = sprintf('%s_%s_%s_%d',database,folder(inds(1)+1:inds(2)-1),folder(inds(2)+1:inds(3)-1),int32(MOS(num_file-2)));
        blocknames{sampleNum} = name;
        
        sampleNum=sampleNum+1;
    end
    
end
end



