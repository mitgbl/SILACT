clear;clc;close all

modelFolder = 'C:\Users\Quadro\MIT 3D Optics Group Dropbox\Deep-learning_ODT\Pred_low_freq';

x_gt = [];
x_pred = [];
index_medium = 1.337;
index_max = 1.42;
dx = 80/383;%um
dy = 80/383;
dz = 80/383*index_medium;
xx = [-127:128].*dx;
yy = [-127:128].*dy;
zz = [-49:50].*dz;

%  for k = 15:19
% % 
% % % load D:\Baoliang\beads\GT_2\GT_2_3_3.mat;
% str1 = 'C:\Users\gbaol\Dropbox (MIT)\2019Summer\Machine learning compressed tomography\RBC_Dataset\GT_BPM';
% % path =  dir(str1);
% 
% Filename_1 = [str1,'\RI_BPM_',num2str(81+k),'.mat'];
% 
% load(Filename_1);
% 
% % index_medium = 1.337;
% n=n_pred;
% 
% figure(1); subplot(221); imagesc(xx,yy,squeeze(n(:,:,50)),[index_medium,index_max]), colorbar, colormap jet, axis equal,  title('x-y');
%         subplot(222); imagesc(xx,zz,squeeze(n(:,128,:))',[index_medium,index_max]), colorbar, colormap jet, axis equal,  title('x-z');
%         subplot(223); imagesc(yy,zz,squeeze(n(128,:,:))',[index_medium,index_max]), colorbar, colormap jet, axis equal,  title('y-z');
%         
% % print(['D:\HEK_cell_training_set\GTs\Unet_output_1\GT_',num2str(k),'.jpg'],'-djpeg','-r600');
% 
% n_pred_1 = n_pred - index_medium;
% phi_cell = n_pred_1((n_pred_1)>0.0005);
% x_gt = [x_gt,phi_cell'];
%         
% Filename_3 = ['C:\Users\gbaol\Dropbox (MIT)\2019Summer\Machine learning compressed tomography\RBC_Dataset\Unet_output\prediction',num2str(k),'.mat'];
% % Filename_3 = 'D:\HEK_cell_training_set\rec_HEK_input_regularized_NPCC.mat';
% load(Filename_3);
% 
% phi_pred = squeeze(n_pred);
% phi_pred = permute(phi_pred,[2 3 1]);
% % L = size(phi_pred,1);
% % for l = 1:L
% %     Phi_pred(:,:,l) = squeeze(phi_pred(l,:,:));
% % end
% y_cell = phi_pred((n_pred_1)>0.0005);
% x_pred = [x_pred,y_cell'];
%       
%  end

load([modelFolder,'\Linear_coef']);
 
%  A = polyfit(x_pred,x_gt,1);

tic
 
 for k = 1:1000
  
% Filename_3 = ['C:\Users\Quadro\MIT 3D Optics Group Dropbox\Deep-learning_ODT\time_lapse\T_cell\Prediction',num2str(k),'.mat'];
% 
% load(Filename_3);
% pred_1 = squeeze(n_pred);
% % pred_1 = permute(pred_1,[2 3 1]);
% % L = size(pred_1,1);
% % for l = 1:L
% %     Pred_1(:,:,l) = squeeze(pred_1(l,:,:));
% % end
% clear n_pred
% n_pred = (pred_1).*A(1)+A(2);
% n_pred = squeeze(permute(n_pred,[2,3,1]));

load(['C:\Users\Quadro\MIT 3D Optics Group Dropbox\Deep-learning_ODT\time_lapse\T_cell_RI\prediction_2_',num2str(k),'.mat'],'n_pred');

index_medium = 1.337;
n = n_pred;
% n=squeeze(permute(phi_pred,[2,3,1]));

figure(2); subplot(221); imagesc(xx,yy,squeeze(n(:,:,50)),[index_medium,index_max]), colorbar, colormap jet, axis equal,title('x-y');
        subplot(222); imagesc(xx,zz,squeeze(n(:,128,:))',[index_medium,index_max]), colorbar, colormap jet, axis equal,title('x-z');
        subplot(223); imagesc(yy,zz,squeeze(n(128,:,:))',[index_medium,index_max]), colorbar, colormap jet, axis equal,title('y-z');
     
 print(['C:\Users\Quadro\MIT 3D Optics Group Dropbox\Deep-learning_ODT\time_lapse\T_cell_RI\pred_',num2str(k),'.jpg'],'-djpeg','-r600');
 
 end
 
 toc