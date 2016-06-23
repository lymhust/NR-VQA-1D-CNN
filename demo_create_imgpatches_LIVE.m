clc;clear;close all;

channel = 1;
FNAMES={'bikes','buildings','buildings2','caps','carnivaldolls',...
    'cemetry','churchandcapitol','coinsinfountain','dancers','flowersonih35'...
    'house','lighthouse','lighthouse2','manfishing','monarch','ocean','paintedhouse'...
    'parrots','plane','rapids','sail1','sail2','sail3','sail4','statue','stream',...
    'studentsculpture','woman','womanhat'};
Fname = {'JPG2K','JPG','GWN','GB','FF'};
imgnum = length(FNAMES);
samplesize = 32;
traindatanum = 10000;
traindata_perfolder = ceil(traindatanum/imgnum);

for t = 1:length(Fname)
    foldername = Fname{t};% Choose the distortion type
    folder = sprintf('.\\%s\\',foldername);
    folder_matfile = sprintf('.\\%s\\matfiles\\',foldername);
    imindex = randperm(imgnum);
    
    TrainImg = [];
    TrainLabels = [];
    TrainScores = [];
    TrainNames = [];
    
    for i = 1:imgnum 
        disp(i);
        str = FNAMES{imindex(i)};
        load(strcat(folder_matfile,str));
        MOS = cell2mat(datas(1,:));
        cls = resetscores(MOS);
        
        folder_img = sprintf('.\\%s\\%s\\',foldername,str);
        [imblock,blocklabel,blockscores,blocknames] = patch_extraction(traindata_perfolder,samplesize,folder_img,MOS,channel,'L');
        TrainImg = [TrainImg imblock];
        TrainLabels = [TrainLabels;blocklabel];
        TrainScores = [TrainScores;blockscores];
        TrainNames = [TrainNames;blocknames];
        
    end
    ind1 = randperm(traindatanum);
    TrainImg = TrainImg(:,ind1);
    TrainLabels = TrainLabels(ind1);
    TrainScores = TrainScores(ind1);
    TrainNames = TrainNames(ind1);
    save(sprintf('.\\Features_CFL\\ImgPatch_LIVE_%s_%d.mat',foldername,samplesize), 'TrainImg', 'TrainLabels','TrainScores','TrainNames');
end



