clear;clc;close all

modelFolder = 'C:\Users\gbaol\Dropbox (MIT)\2019Summer\Machine learning compressed tomography\Model\RBC_Dataset\GT_BPM';
dataFolder = 'C:\Users\gbaol\Dropbox (MIT)\2019Summer\Machine learning compressed tomography\Model\RBC_Dataset\3T3_model_output_1';

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

 for k = 1:19

    gtFile = [modelFolder,'\RI_BPM_',num2str(k),'.mat'];
    predFile = [dataFolder,'\Prediction',num2str(k),'.mat'];

    if exist(gtFile) == 0 || exist(predFile) == 0
        continue
    else
    end

    load(gtFile);
%     gt = permute(gt,[2 3 1]);
%     gt = n_pred;
    gt = squeeze(n_pred);
%     gt = gt./100+index_medium;
%     gt = permute(gt,[2 3 1]);
    n=gt;

    figure(22); subplot(221); imagesc(xx,yy,squeeze(n(:,:,50)),[index_medium,index_max]), colorbar, colormap jet, axis equal,  title('x-y');
        subplot(222); imagesc(xx,zz,squeeze(n(:,128,:))',[index_medium,index_max]), colorbar, colormap jet, axis equal,  title('x-z');
        subplot(223); imagesc(yy,zz,squeeze(n(128,:,:))',[index_medium,index_max]), colorbar, colormap jet, axis equal,  title('y-z');
        
%     print([figFolder,'\GT_',num2str(k),'.jpg'],'-djpeg','-r600');
    % save(['C:\Users\gbaol\Dropbox (MIT)\2019Summer\Machine learning compressed tomography\Testing_Results\Unet_output_3\GT_',num2str(k),'.mat'],'n_pred');
    % 
    n_pred_1 = gt - index_medium;
    phi_cell = gt((n_pred_1)>0.005);
    x_gt = [x_gt,phi_cell'];
    %         
    load(predFile);
    % 
    phi_pred = squeeze(n_pred);
    phi_pred = permute(phi_pred,[2 3 1]);

    figure(55); subplot(221); imagesc(xx,yy,squeeze(phi_pred(:,:,50))), colorbar, colormap jet, axis equal,  title('x-y');
        subplot(222); imagesc(xx,zz,squeeze(phi_pred(:,128,:))'), colorbar, colormap jet, axis equal,  title('x-z');
        subplot(223); imagesc(yy,zz,squeeze(phi_pred(128,:,:))'), colorbar, colormap jet, axis equal,  title('y-z');
        
    % % L = size(phi_pred,1);
    % % for l = 1:L
    % %     Phi_pred(:,:,l) = squeeze(phi_pred(l,:,:));
    % % end
    y_cell = phi_pred((n_pred_1)>0.005);
    x_pred = [x_pred,y_cell'];
      
 end
% load([modelFolder,'\Linear_coef']);
 
 A = polyfit(x_pred,x_gt,1);
 
 for k = 1:99
  
Filename_3 = [dataFolder,'\Prediction',num2str(k),'.mat'];

load(Filename_3);
pred_1 = squeeze(n_pred);
pred_1 = permute(pred_1,[2 3 1]);
% L = size(pred_1,1);
% for l = 1:L
%     Pred_1(:,:,l) = squeeze(pred_1(l,:,:));
% end
phi_pred = (pred_1).*A(1)+A(2);

save([dataFolder,'\prediction_2_',num2str(k),'.mat'],'phi_pred');

% index_medium = 1.337;
% n=squeeze(permute(phi_pred,[2,3,1]));
n=squeeze(phi_pred);

figure(2); subplot(221); imagesc(xx,yy,squeeze(n(:,:,50)),[index_medium,index_max]), colorbar, colormap jet, axis equal,title('x-y');
        subplot(222); imagesc(xx,zz,squeeze(n(:,128,:))',[index_medium,index_max]), colorbar, colormap jet, axis equal,title('x-z');
        subplot(223); imagesc(yy,zz,squeeze(n(128,:,:))',[index_medium,index_max]), colorbar, colormap jet, axis equal,title('y-z');
     
%  print([dataFolder,'\pred_',num2str(k),'.jpg'],'-djpeg','-r600');
 
 end