clear;clc;close all

rootFolder = 'C:\Users\gbaol\Dropbox (MIT)\2019Summer\Machine learning compressed tomography\Model\RBC_Dataset';
dataFolder1 = [rootFolder,'\Time_lapse_cells_RBC_model'];
dataFolder2 = [rootFolder,'\Time_lapse_cells_3T3_model'];
figFolder = [rootFolder,'\Figure_single_cell]'];
if exist(figFolder) == 0
    mkdir(figFolder);
else
end
paths = dir([dataFolder1,'\*.mat']);
% paths = dir([dataFolder,'\*.mat']);
LL = length(paths);

index_medium = 1.337;
index_max = 1.44;
dx = 80/383;%um
dy = 80/383;
dz = 80/383;
xx = [-127:128].*dx;
yy = [-127:128].*dy;
zz = [-49:50].*dz;
alpha = 0.2;

for k = 1:LL
    
%     k
    
    cellFile1 = [dataFolder1,'\RBC_1_frame_',num2str(k),'.mat'];
    load(cellFile1);
    
    n_rbc_1 = n_rbc;
    
    cellFile2 = [dataFolder2,'\RBC_1_frame_',num2str(k),'.mat'];
    load(cellFile2);
    
    n_rbc_2 = n_rbc;
        
    n_xy_1 = squeeze(max(n_rbc_1,[],3));
    n_xz_1 = squeeze(max(n_rbc_1,[],2));
    n_yz_1 = squeeze(max(n_rbc_1,[],1));
    
    figure(555); subplot(221); imagesc(xx,yy,n_xy_1,[index_medium,index_max]), colorbar, colormap jet, axis equal,  title('x-y');
            subplot(222); imagesc(xx,zz,n_xz_1,[index_medium,index_max]), colorbar, colormap jet, axis equal,  title('x-z');
            subplot(223); imagesc(yy,zz,n_yz_1,[index_medium,index_max]), colorbar, colormap jet, axis equal,  title('y-z'); 
            
    BW_1 = zeros(size(n_rbc_1));
    BW_1(n_rbc_1 > 1.35) = 1;
    s1 = regionprops3(BW_1, "Volume", "SurfaceArea","EigenValues");
    
    eg_length_1 = s1.EigenValues{1};
    disp(eg_length_1);
    
    long_1 = eg_length_1(1);
    short_1 = mean(eg_length_1(2:3));
    
    ec_3d_1(k) = long_1/short_1;
%             
    n_xy_2 = squeeze(max(n_rbc_2,[],3));
    n_xz_2 = squeeze(max(n_rbc_2,[],2));
    n_yz_2 = squeeze(max(n_rbc_2,[],1));
    
    figure(556); subplot(221); imagesc(xx,yy,n_xy_2,[index_medium,index_max]), colorbar, colormap jet, axis equal,  title('x-y');
            subplot(222); imagesc(xx,zz,n_xz_2,[index_medium,index_max]), colorbar, colormap jet, axis equal,  title('x-z');
            subplot(223); imagesc(yy,zz,n_yz_2,[index_medium,index_max]), colorbar, colormap jet, axis equal,  title('y-z'); 
            
    BW_2 = zeros(size(n_rbc_2));
    BW_2(n_rbc_2 > 1.35) = 1;
    s2 = regionprops3(BW_2, "Volume", "SurfaceArea","EigenValues");
    
    eg_length_2 = s2.EigenValues{1};
    disp(eg_length_2);
    
    long_2 = eg_length_2(1);
    short_2 = mean(eg_length_2(2:3));
    
    ec_3d_2(k) = long_2/short_2;           
            
end

xx = 1:LL;

figure(111);
% subplot(221);
scatter(xx./10,ec_3d_1,15,'filled');hold on
scatter(xx./10,ec_3d_2,15,'filled');hold on
% scatter(xx./10,ec_xz,15,'filled');hold on
% scatter(xx./10,ec_yz,15,'filled');hold on
% errorbar(xx-0.15,l1_mean(:,1),l1_std(:,1),'k','linewidth',1.2,'linestyle','none');
% errorbar(xx+0.15,l1_mean(:,2),l1_std(:,2),'k','linewidth',1.2,'linestyle','none');
% xticklabels({'4','8','16','24','32'});
axis([0 2.7  2 7]);
% yticks([0,0.2,0.4,0.6,0.8,1]);
% yticklabels({'0.5','0.7','1'});
xlabel('time (ms)');
ylabel('3D eccentricity');
% set(gca,'box','off') 
set(get(gca,'xlabel'),...'fontangle','italic','fontweight','bold',...
'fontsize',15)
set(get(gca,'ylabel'),...'fontangle','italic','fontweight','bold',...
'fontsize',15)
set(get(gca,'xaxis'),...'fontangle','italic','fontweight','bold',...
'fontsize',15)
set(get(gca,'yaxis'),...'fontangle','italic','fontweight','bold',...
'fontsize',15)
%set(get(gca,'title'),'fontangle','italic','fontweight','bold',...
%'fontsize',15)
h = legend('RBC trained model','3T3 trained model');
set(h,...'fontangle','italic','fontweight','bold',...
'fontsize',15,'location','best')

print([figFolder,'\eccentricity_3d_both_model.jpg'],'-djpeg','-r600');

figure(112);
% subplot(221);
scatter(xx./10,ec_3d_1,15,'filled');hold on
% scatter(xx./10,ec_3d_2,15,'filled');hold on
% scatter(xx./10,ec_xz,15,'filled');hold on
% scatter(xx./10,ec_yz,15,'filled');hold on
% errorbar(xx-0.15,l1_mean(:,1),l1_std(:,1),'k','linewidth',1.2,'linestyle','none');
% errorbar(xx+0.15,l1_mean(:,2),l1_std(:,2),'k','linewidth',1.2,'linestyle','none');
% xticklabels({'4','8','16','24','32'});
axis([0 2.7  2 8]);
% yticks([0,0.2,0.4,0.6,0.8,1]);
% yticklabels({'0.5','0.7','1'});
xlabel('Time (ms)');
ylabel('3D eccentricity');
% set(gca,'box','off') 
set(get(gca,'xlabel'),...'fontangle','italic','fontweight','bold',...
'fontsize',12)
set(get(gca,'ylabel'),...'fontangle','italic','fontweight','bold',...
'fontsize',12)
set(get(gca,'xaxis'),...'fontangle','italic','fontweight','bold',...
'fontsize',12)
set(get(gca,'yaxis'),...'fontangle','italic','fontweight','bold',...
'fontsize',12)
%set(get(gca,'title'),'fontangle','italic','fontweight','bold',...
%'fontsize',15)
% h = legend('RBC trained model','3T3 trained model');
% set(h,...'fontangle','italic','fontweight','bold',...
% 'fontsize',15,'location','best')

print([figFolder,'\eccentricity_3d_maintext.jpg'],'-djpeg','-r600');
% 
% c1 = [0, 114, 189]./255;
% c2 = [217, 83, 25]./255;
% c3 = [237, 177, 32]./255;
% c4 = [126, 47, 142]./255;
% c5 = [119, 172, 48]./255;
% c6 = [77, 190, 238]./255;
% c7 = [162, 20, 47]./255;
% 
% figure(112);
% % subplot(221);
% scatter(xx./10,volume,15,c1,'filled');hold on
% axis([0 2.7 0 120]);
% % scatter(xx,ec_xz,15,'filled');hold on
% % scatter(xx,ec_yz,15,'filled');hold on
% % errorbar(xx-0.15,l1_mean(:,1),l1_std(:,1),'k','linewidth',1.2,'linestyle','none');
% % errorbar(xx+0.15,l1_mean(:,2),l1_std(:,2),'k','linewidth',1.2,'linestyle','none');
% % xticklabels({'4','8','16','24','32'});
% % yticks([0,0.2,0.4,0.6,0.8,1]);
% % yticklabels({'0.5','0.7','1'});
% xlabel('time(ms)');
% ylabel('volume(\mum^3)');
% set(gca,'box','off') 
% set(get(gca,'xlabel'),'fontangle','italic','fontweight','bold',...
% 'fontsize',15)
% set(get(gca,'ylabel'),'fontangle','italic','fontweight','bold',...
% 'fontsize',15)
% set(get(gca,'xaxis'),'fontangle','italic','fontweight','bold',...
% 'fontsize',15)
% set(get(gca,'yaxis'),'fontangle','italic','fontweight','bold',...
% 'fontsize',15)
% set(get(gca,'title'),'fontangle','italic','fontweight','bold',...
% 'fontsize',15)
% % h = legend('EC_x_y','EC_x_z','EC_y_z');
% % set(h,'fontangle','italic','fontweight','bold',...
% % 'fontsize',12,'location','east')
% 
% print([figFolder,'\RBC_volume_2.jpg'],'-djpeg','-r600');
% 
% figure(113);
% % subplot(221);
% scatter(xx./10,mass,15,c2,'filled');hold on
% axis([0 2.7 0 25]);
% % scatter(xx,ec_xz,15,'filled');hold on
% % scatter(xx,ec_yz,15,'filled');hold on
% % errorbar(xx-0.15,l1_mean(:,1),l1_std(:,1),'k','linewidth',1.2,'linestyle','none');
% % errorbar(xx+0.15,l1_mean(:,2),l1_std(:,2),'k','linewidth',1.2,'linestyle','none');
% % xticklabels({'4','8','16','24','32'});
% % yticks([0,0.2,0.4,0.6,0.8,1]);
% % yticklabels({'0.5','0.7','1'});
% xlabel('time(ms)');
% ylabel('dry mass(pg)');
% set(gca,'box','off') 
% set(get(gca,'xlabel'),'fontangle','italic','fontweight','bold',...
% 'fontsize',15)
% set(get(gca,'ylabel'),'fontangle','italic','fontweight','bold',...
% 'fontsize',15)
% set(get(gca,'xaxis'),'fontangle','italic','fontweight','bold',...
% 'fontsize',15)
% set(get(gca,'yaxis'),'fontangle','italic','fontweight','bold',...
% 'fontsize',15)
% set(get(gca,'title'),'fontangle','italic','fontweight','bold',...
% 'fontsize',15)
% 
% print([figFolder,'\RBC_dry_mass_2.jpg'],'-djpeg','-r600');