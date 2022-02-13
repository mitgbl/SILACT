clear;clc;close all

str0 = 'C:\Users\gbaol\Dropbox (MIT)\2020Spring\Deep-learning ODT\multiplex_training';
str1 = [str0,'\GT_BPM'];
str2 = [str0,'\GT_Rytov'];
str3 = [str0,'\inputs_1'];
str4 = [str0,'\inputs_2'];

% images = []
% imgs = [3:31,34:60]
    
filename_1 = [str2,'\RI_Rytov_12.mat'];

load(filename_1);

xx = (1:256);
yy = (1:256);
zz = (1:100);

index_medium = 1.337;

figure(555); subplot(221); imagesc(xx,yy,squeeze(n_pred(:,:,50)),[index_medium,1.42]), colorbar, colormap jet, axis equal, axis off,title('x-y');
        subplot(222); imagesc(xx,zz,squeeze(n_pred(:,128,:))',[index_medium,1.42]), colorbar, colormap jet, axis equal, axis off,title('x-z');
        subplot(223); imagesc(yy,zz,squeeze(n_pred(128,:,:))',[index_medium,1.42]), colorbar, colormap jet, axis equal, axis off,title('y-z');

filename_2 = [str1,'\RI_BPM_12.mat'];

load(filename_2);

xx = (1:256);
yy = (1:256);
zz = (1:100).*1.337;

index_medium = 1.337;

figure(556); subplot(221); imagesc(xx,yy,squeeze(n_pred(:,:,50)),[index_medium,1.42]), colorbar, colormap jet, axis equal, axis off,title('x-y');
        subplot(222); imagesc(xx,zz,squeeze(n_pred(:,128,:))',[index_medium,1.42]), colorbar, colormap jet, axis equal, axis off,title('x-z');
        subplot(223); imagesc(yy,zz,squeeze(n_pred(128,:,:))',[index_medium,1.42]), colorbar, colormap jet, axis equal, axis off,title('y-z');
      