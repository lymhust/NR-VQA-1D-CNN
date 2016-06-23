function str = measurement(Trueval, Preval)
CC = corrcoef(Trueval,Preval);
SROCC = corr(Trueval,Preval, 'type' , 'Spearman');
AE = abs(Trueval - Preval); MAE = mean(AE); stdAE = std(AE); 
RE = AE ./ Trueval; MRE = mean(RE) * 100; stdRE = std(RE) * 100;
MSE = mean((Trueval - Preval).^2);
str = sprintf('CC=%.4f,SROCC=%.4f,AE=%.2f¡À%.2f,RE=%.2f¡À%.2f,MSE=%.2f',CC(1,2),SROCC,MAE,stdAE,MRE,stdRE,MSE);