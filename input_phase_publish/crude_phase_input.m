clear;clc;close all

str0 = 'C:\Users\gbaol\Dropbox (MIT)\20201030_both_HeLa';
mkdir([str0,'\Inputs']);
mkdir([str0,'\Input_views']);

paths = dir(str0);
L = size(paths);

for m = 3:L
str1 = [str0,'\',paths(m).name,'\sample_m'];
str2 = [str0,'\',paths(3).name,'\bg_m'];
% str3 = [str0,'\',paths(m).name,'\sample_s'];
% str4 = [str0,'\',paths(m).name,'\bg_s'];
% mkdir([str0,'\',paths(m).name,'\crude_phases_0.5_2']);

% if exist([str0,'\',paths(m).name,'\crude_phases_0.5_2\phase_crude_0.5_2.mat']) ~= 0
%     continue
% else
% end
% 
% mkdir([str0,'\',paths(m).name,'\crude_phases_0.5_2']);

lampta=0.532; % wavelength of laser/um
k_0=2*pi/lampta; % wave number
NA = 1.4;
delta_x = 20;
delta_y = 20;
NN = 256;
lambda0 = 532e-9; % free space wavelength (m)
magnification = 383; % Mnominal*Fused/Fnominal

%% Angle-multiplexing phase retrieval

paths_s = dir(str1);
paths_bg = dir(str2);
L = size(paths_s);

frames = 1:13;

filename_0 = [str2,'\',paths_bg(3).name];
Im_0 = imread(filename_0);
Im_0 = im2double(Im_0(:,:,1));

[mi,mj,mask_1]=phase_recover_normal(NA,k_0,Im_0,delta_x/magnification,delta_y/magnification);
% 

        for l = 2:(L-2)
            
            Filename_1 = [str1,'\',paths_s(l+2).name];
            Im_1 = imread(Filename_1);
            Im_1 = Im_1(:,:,1);
            I_multi = im2double(Im_1);
            
            [C_sub_1,C_sub_2,C_sub_3,C_sub_4] = phase_recover_multiplex_2(mi,mj,mask_1,I_multi,NA*0.5,k_0,delta_x/magnification);
            Phi_1 = (angle(C_sub_1));
            Phi_2 = (angle(C_sub_2));
            Phi_3 = (angle(C_sub_3));
            Phi_4 = (angle(C_sub_4));
%             C_multi = phase_recover_circle(mi,mj,mask_1,I_multi);
            
            Filename_2 = [str2,'\',paths_bg(l+2).name];
            Im_2 = imread(Filename_2);
            Im_2 = Im_2(:,:,1);
            I_multi_bg = im2double(Im_2);
                        
            [C_bg_1,C_bg_2,C_bg_3,C_bg_4] = phase_recover_multiplex_2(mi,mj,mask_1,I_multi_bg,NA*0.5,k_0,delta_x/magnification);
            Phi_bg_1 = (angle(C_bg_1));
            Phi_bg_2 = (angle(C_bg_2));
            Phi_bg_3 = (angle(C_bg_3));
            Phi_bg_4 = (angle(C_bg_4));
            
             Phi1 = unwrap2(Phi_1-Phi_bg_1);
             [Phi1] = RemoveTilt_phi(100,100,Phi1);
            Phi2 = unwrap2(Phi_2-Phi_bg_2);
            [Phi2] = RemoveTilt_phi(100,100,Phi2);
             Phi3 = unwrap2(Phi_3-Phi_bg_3);
            [Phi3] = RemoveTilt_phi(100,100,Phi3);
            Phi4 = unwrap2(Phi_4-Phi_bg_4);
            [Phi4] = RemoveTilt_phi(100,100,Phi4);
            
%             Phi_bg_1 = unwrap2(Phi_bg_1);
%             Phi_bg_2 = unwrap2(Phi_bg_2);
%             Phi_bg_3 = unwrap2(Phi_bg_3);
%             Phi_bg_4 = unwrap2(Phi_bg_4);
%             
            Phi_stack(:,:,(l-2)*4+2) = imresize(Phi1,[256,256]);
            Phi_stack(:,:,(l-2)*4+3) = imresize(Phi2,[256,256]);
            Phi_stack(:,:,(l-2)*4+4) = imresize(Phi3,[256,256]);
            Phi_stack(:,:,(l-2)*4+5) = imresize(Phi4,[256,256]);
            
            figure(666);subplot(221);imagesc(Phi1,[-2*pi,2*pi]);colormap jet;colorbar
            subplot(222);imagesc(Phi2,[-2*pi,2*pi]);colormap jet;colorbar
            subplot(223);imagesc(Phi3,[-2*pi,2*pi]);colormap jet;colorbar
            subplot(224);imagesc(Phi4,[-2*pi,2*pi]);colormap jet;colorbar
            
%             figure(777);subplot(221);imagesc(squeeze(Phi_stack(:,:,(l-2)+3)),[-2*pi,2*pi]);colormap jet;colorbar
%             subplot(222);imagesc(squeeze(Phi_stack(:,:,(l-2)+15)),[-2*pi,2*pi]);colormap jet;colorbar
%             subplot(223);imagesc(squeeze(Phi_stack(:,:,(l-2)+27)),[-2*pi,2*pi]);colormap jet;colorbar
%             subplot(224);imagesc(squeeze(Phi_stack(:,:,(l-2)+39)),[-2*pi,2*pi]);colormap jet;colorbar
            
%             figure(888);subplot(121);imagesc(abs(C_multi));colormap jet;colorbar
%             subplot(122);imagesc(angle(C_multi));colormap jet;colorbar
%             
%             figure(999);subplot(121);imagesc(abs(C_multi_bg));colormap jet;colorbar
%             subplot(122);imagesc(angle(C_multi_bg));colormap jet;colorbar
            
%              print([str0,'\',paths(m).name,'\crude_phases_0.5_2\phase_',num2str(l),'.jpg'],'-djpeg','-r600');
             
%              Phi_multi = imresize(angle(C_multi),[512,512]);
%              G_out(:,:,l-1) = exp(1i*Phi_multi);
%              
%              Phi_bg_multi = imresize(angle(C_multi_bg),[512,512]);
%              G_in(:,:,l-1) = exp(1i*Phi_bg_multi);
                      
        end
        
%         save([str0,'\',paths(m).name,'\crude_phases_0.5_2\phase_crude_0.5_2.mat'],'Phi_stack');
        
       Phi_sub = Phi_stack(:,:,38:41);
        
        for l = 1:4
            Phi_crude(:,:,l) = imresize(squeeze(Phi_sub(:,:,l)),[256,256]);
        end
        
        save([str0,'\Inputs\Input_',num2str(m-2),'.mat'],'Phi_crude');
        
        figure(999);
        subplot(221);imagesc(squeeze(Phi_crude(:,:,1)),[-2*pi,2*pi]);colormap jet;colorbar
        subplot(222);imagesc(squeeze(Phi_crude(:,:,2)),[-2*pi,2*pi]);colormap jet;colorbar
        subplot(223);imagesc(squeeze(Phi_crude(:,:,3)),[-2*pi,2*pi]);colormap jet;colorbar
        subplot(224);imagesc(squeeze(Phi_crude(:,:,4)),[-2*pi,2*pi]);colormap jet;colorbar
    
      print([str0,'\Input_views\Input_',num2str(m-2),'.jpg'],'-djpeg','-r600');
        
end
        
