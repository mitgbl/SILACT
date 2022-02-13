clear;clc;close all
%% parameters
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

%% paths
str1 = 'C:\Users\gbaol\Dropbox (MIT)\2019Summer\Machine learning compressed tomography\Model\Pred_crude_new_Pytorch_2';
% dataFolder = [str1,'\Unet_Output_RBC'];
% predFolder = [str1,'\Pred_RBC'];
% figFolder = [str1,'\Figures_RBC'];
% evalFolder = [str1,'\Evals'];
% if exist(predFolder) == 0
%     mkdir(predFolder);
% else
% end
% if exist(figFolder) == 0
%     mkdir(figFolder);
% else
% end
% if exist(evalFolder) == 0
%     mkdir(evalFolder);
% else
% end

 for k = 1:38

    gtFile = [str1,'\GT_',num2str(k),'.mat'];
    predFile = [str1,'\Prediction',num2str(k),'.mat'];

    if exist(gtFile) == 0 || exist(predFile) == 0
        continue
    else
    end

    load(gtFile);
%     gt = permute(gt,[2 3 1]);
%     gt = n_pred;
    gt = squeeze(gt);
    gt = gt./100+index_medium;
    gt = permute(gt,[2 3 1]);
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
    phi_pred = squeeze(phi_pred);
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
 
 A = polyfit(x_pred,x_gt,1);
%  
 save([str1,'\Linear_coef'],'A');