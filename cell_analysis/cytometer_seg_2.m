clear;clc;close all

rootFolder = 'D:\Baoliang\Flow_3t3_pred';
dataFolder = [rootFolder,'\flow_rate_50_exp_time_15us_pred'];
paths = dir([dataFolder,'\*.mat']);
LL = length(paths);
cellFolder = [rootFolder,'\flow_rate_50_exp_time_15us_cells_1'];
if exist(cellFolder) == 0
    mkdir(cellFolder);
else
end
idx = 0;
% k = 171;

thres = 1.37;

index_medium = 1.337;
index_max = 1.42;
dx = 80/383;%um
dy = 80/383;
dz = 80/383*index_medium;
xx = [-127:128].*dx;
yy = [-127:128].*dy;
zz = [-49:50].*dz;

for k = 1:LL
    
%     if mod(k,3)~=1
%         continue
%     else
%     end

    predFile = [dataFolder,'\prediction_2_',num2str(k),'.mat'];
    load(predFile,'n_pred');
    n_pred = squeeze(permute(n_pred,[2,3,1]));

%     n_pred = permute(phi_pred,[2 3 1]);
    for m = 1:100
        n_2d = n_pred(:,:,m);
        n_pred_1(:,:,m) = imresize(n_2d,[256,256]);
    end

%     figure(555); subplot(221); imagesc(xx,yy,squeeze(n_pred_1(:,:,50)),[index_medium,index_max]), colorbar, colormap jet, axis equal,  title('x-y');
%             subplot(222); imagesc(xx,zz,squeeze(n_pred_1(:,128,:))',[index_medium,index_max]), colorbar, colormap jet, axis equal,  title('x-z');
%             subplot(223); imagesc(yy,zz,squeeze(n_pred_1(128,:,:))',[index_medium,index_max]), colorbar, colormap jet, axis equal,  title('y-z'); 

    n_xy = squeeze(n_pred_1(:,:,50));
    BW = zeros(256,256);
    BW(n_xy>thres) = 1;
    Ind = bwlabel(BW,4);

%     figure(111);imagesc(Ind);colormap jet;colorbar

    L = size(n_pred_1,3);
    N  = max(Ind(:));

    for ii = 1:N
        lb =zeros(256,256);
        lb(Ind==ii) = 1;
%     idx = 0;
        if (sum(lb(:))<500)
            continue
        else
            idx = idx+1;
        end
        for jj = 1:L
            lb_3d(:,:,jj) = lb;
        end
%    lb_2d(:,:,ii) = lb;
        lb_3d(n_pred_1<thres) = 0;
        LG(:,:,:,ii) = lb_3d;
        n_cell(:,:,:,idx) = n_pred_1.*lb_3d;
    
        n_pred_2 = squeeze(n_cell(:,:,:,idx));

%         figure(555); subplot(221); imagesc(xx,yy,squeeze(n_pred_2(:,:,50)),[index_medium,index_max]), colorbar, colormap jet, axis equal,  title('x-y');
%             subplot(222); imagesc(xx,zz,squeeze(n_pred_2(:,128,:))',[index_medium,index_max]), colorbar, colormap jet, axis equal,  title('x-z');
%             subplot(223); imagesc(yy,zz,squeeze(n_pred_2(128,:,:))',[index_medium,index_max]), colorbar, colormap jet, axis equal,  title('y-z'); 
    end

end

n_cell = single(n_cell);
save([cellFolder,'\flow_rate_50_exp_time_15us_cells_2.mat'],'n_cell','-v7.3');
