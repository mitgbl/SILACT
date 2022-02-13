clear;clc;close all

% dataFolder = 'D:\Baoliang\Flow_3t3_pred\flow_rate_50_exp_time_15us_cells_1';
dataFolder = 'C:\Users\Quadro\MIT 3D Optics Group Dropbox\Deep-learning_ODT\20210812_jurket_C001H001S0003\cells';
% cellFile = [dataFolder,'\flow_rate_40_exp_time_15us_cells_3.mat'];
% 
% load(cellFile);
% LL = size(n_cell,4);
thres = 1.37;

index_medium = 1.337;
index_max = 1.42;
dx = 80/383;%um
dy = 80/383;
dz = 80/383*index_medium;
xx = [-127:128].*dx;
yy = [-127:128].*dy;
zz = [-49:50].*dz;
idx = 0;
index_medium = 1.337;
alpha = 0.2;

% for l = 1:3
    
%     cellFile = [dataFolder,'\flow_rate_40_exp_time_15us_cells_',num2str(l),'.mat'];
    cellFile = [dataFolder,'\flow_rate_50_T_cell.mat'];
    load(cellFile);
    LL = size(n_cell,4);

for k = 1:LL
    
%     idx = 0;
    n_c = squeeze(n_cell(:,:,:,k));
%     volume = length(n_c(n_c>1.37)).*(dx^3);
    
    n_x1 = squeeze(n_c(:,1,50));
    n_x2 = squeeze(n_c(:,256,50));
    n_y1 = squeeze(n_c(1,:,50));
    n_y2 = squeeze(n_c(256,:,50));
    
    ri_x1 = max(n_x1);
    ri_x2 = max(n_x2);
    ri_y1 = max(n_y1);
    ri_y2 = max(n_y2);
    
    if ri_x1>1.36 || ri_x2>1.36 || ri_y1>1.36 || ri_y2>1.36 ||(length(n_c(n_c>1.36))*(dx^3))>5000 || (length(n_c(n_c>1.36))*(dx^3))<100                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     00
        continue
    else
        %idx = idx + 1;
    end
    
    figure(555); subplot(221); imagesc(xx,yy,squeeze(n_c(:,:,50)),[index_medium,index_max]), colorbar, colormap jet, axis equal,  title('x-y');
         subplot(222); imagesc(xx,zz,squeeze(n_c(:,128,:))',[index_medium,index_max]), colorbar, colormap jet, axis equal,  title('x-z');
         subplot(223); imagesc(yy,zz,squeeze(n_c(128,:,:))',[index_medium,index_max]), colorbar, colormap jet, axis equal,  title('y-z'); 
    
%     n_s = n_c;
%     n_s(abs(n_c-1.37)>0.01) = index_medium;
%     
%     figure(556); subplot(221); imagesc(xx,yy,squeeze(n_s(:,:,50)),[index_medium,index_max]), colorbar, colormap jet, axis equal,  title('x-y');
%          subplot(222); imagesc(xx,zz,squeeze(n_s(:,128,:))',[index_medium,index_max]), colorbar, colormap jet, axis equal,  title('x-z');
%          subplot(223); imagesc(yy,zz,squeeze(n_s(128,:,:))',[index_medium,index_max]), colorbar, colormap jet, axis equal,  title('y-z'); 
    
%     volume(idx) = length(n_c(n_c>1.37)).*(dx^3);
%     
%     Chi = n_c.^2-index_medium.^2;
%     Chi_mean = mean(Chi(n_c>1.37));
%     mass(idx) = (Chi_mean.*volume(idx))./(2.*index_medium.*alpha);
%     
%     thres = 0.01;
%     A_surface(idx) = length(n_c(abs(n_c-1.37)<0.01)).*(dx^2);
    BW = zeros(size(n_c));
    BW(n_c>1.36) = 1;
    s = regionprops3(BW, "Volume", "SurfaceArea","EigenValues");
    if (s.Volume)*(dx^3)<100
        continue
    else
        idx = idx+1;
    end
    n_mean(idx) = mean(n_c(n_c>1.36));
    volume(idx) = (s.Volume)*(dx^3);
    A_surface(idx) = (s.SurfaceArea)*(dx^2);
    Chi = n_c.^2-index_medium.^2;
    Chi_mean = mean(Chi(n_c>1.36));
    mass(idx) = (Chi_mean.*volume(idx))./(2.*index_medium.*alpha);
    
%     BW_1 = zeros(size(n_c));
%     BW_1(n_c > 1.37) = 1;
%     s1 = regionprops3(BW_1, "Volume", "SurfaceArea","EigenValues");
    
%     eg_length_1 = s.EigenValues{1};
%     disp(eg_length_1);
%     
%     long_1 = eg_length_1(1);
%     short_1 = mean(eg_length_1(2:3));
%     
%     ec_3d(idx) = long_1/short_1;
    
end

% end
% 
figure(111);subplot(221);histogram(volume,40);
subplot(222);histogram(mass,40);
subplot(223);histogram(n_mean,40);
subplot(224);histogram(A_surface,40);

% figure;histogram(ec_3d,40);

save([dataFolder,'\cell_volumes_1.mat'],'volume');
save([dataFolder,'\cell_dry_mass_1.mat'],'mass');
save([dataFolder,'\cell_mean_RI_1.mat'],'n_mean');
save([dataFolder,'\cell_surface_area_1.mat'],'A_surface');
% save([dataFolder,'\cell_eccentricity_5.mat'],'ec_3d');
